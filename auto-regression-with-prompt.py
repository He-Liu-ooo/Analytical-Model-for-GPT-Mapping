import argparse
import random
import math
import copy
from operator import itemgetter
from utils import Stats
from utils import latency

""" 
TODO
1. create stats for utilization evaluation of each die and each PE
2. maybe there are better algorithm for prefill stage distribution and prefill & generation stage distribution, for now they all use the same simple algorithm
3. consider IA loading latency and cross-die communication overhead, present version assumes IA loading latency can be perfectly hidden behind weigght loading latency
4. try to support more partial sum in sub-pair in FC calculation, not only one
5. how many batch in a die also affects the load balance in the long term, can we do sth. with this?
"""

def argparser():
    """ Argument parser. """

    ap = argparse.ArgumentParser()

    # HW
    ap.add_argument('--die-num', type = int, default = 3,
                    help = 'number of dies in the system')
    ap.add_argument('--core-num-in-die', type = int, default = 16,     
                    help = 'number of cores in a die')
    ap.add_argument('--mac-lane', type = int, default = 16)
    ap.add_argument('--mac-num', type = int, default = 32)
    ap.add_argument('--SRAM2_height', type = int, default = 2048,
                    help = 'number of 32B can be stored in a core\'s SRAM2')
    
    # MODEL
    ap.add_argument('--batch-size', type = int, default = 1024,
                    help = 'number of tokens in a big batch')
    ap.add_argument('--embedding-dim', type = int, default = 4096,
                    help = 'embedding dimension')
    ap.add_argument('--head-num', type = int, default = 32,
                    help = 'number of heads in multihead attention')
    ap.add_argument('--decoder-num', type = int, default = 32,
                    help = 'number of decoders in the model')

    # PARAM
    ap.add_argument('--cal-latency', type = int, default = 3,
                    help = 'latency of calculating a mac_num dot production(8bit)')
    ap.add_argument('--psum-vec-cal-latency', type = int, default = 0.5,
                    help = 'latency of adding 2 mac_lane * 3Byte data')
    ap.add_argument('--psum-cube-cal-N-latency', type = int, default = 0.5,
                    help = 'latency of adding N(core number in a die) mac_lane * mac_lane * 3Byte data')
    ap.add_argument('--psum-cube-loading-latency', type = int, default = 0.5,
                    help = 'latency of reading a mac_lane * mac_lane * 3Byte data from HMC')
    ap.add_argument('--weight-loading-latency', type = int, default = 0.5,
                    help = 'latency of transfering mac_num Byte of data from HMC to SRAM of core')
    ap.add_argument('--psum-cube-wb-latency', type = int, default = 0.5,
                    help = 'latency of writing a mac_lane * mac_lane * 3Byte data to HMC')
    
    
    # OTHERS
    ap.add_argument('--debug-flag', type = bool, default = True)       
    ap.add_argument('--sim-mode', type = int, default = 1,
                    help = '0 for my design, 1 for FlexGen prompt/answer padding implementation')
    
    return ap

def dump_res():
    pass

def init_requests_pool():
    requests_pool = []
    
    with open("GPT_requests_pool.txt", "r") as f:
        ss = f.readlines()
        for s in ss:
            s = s.strip('\n')
            s = s.split(',')
            requests_pool.append([False, int(s[0]), int(s[1])])
    
    return requests_pool
        
def requests_done(seq_len_table):
    done = True
    for i in seq_len_table:
        if i[3] == False:
            done = False
            break
    return done

def latency_acc1(stage_latency, Statistics, D):
    Statistics.latency.cal += stage_latency.cal * D
    Statistics.latency.psum_vec_cal += stage_latency.psum_vec_cal * D
    Statistics.latency.psum_cube_cal_N += stage_latency.psum_cube_cal_N * D
    Statistics.latency.psum_cube_loading += stage_latency.psum_cube_loading * D
    Statistics.latency.weight_loading += stage_latency.weight_loading * D
    Statistics.latency.psum_cube_wb += stage_latency.psum_cube_wb * D
    
def latency_acc2(stage_latency, latencys):
    new_latencys = latency()
    
    new_latencys.cal = latencys.cal + stage_latency.cal
    new_latencys.psum_vec_cal = latencys.psum_vec_cal + stage_latency.psum_vec_cal
    new_latencys.psum_cube_cal_N = latencys.psum_cube_cal_N + stage_latency.psum_cube_cal_N
    new_latencys.psum_cube_loading = latencys.psum_cube_loading + stage_latency.psum_cube_loading
    new_latencys.weight_loading = latencys.weight_loading + stage_latency.weight_loading
    new_latencys.psum_cube_wb = latencys.psum_cube_wb + stage_latency.psum_cube_wb
    
    return new_latencys

def calculate_weights_col_assignment(seg_num, weights_maclane_col_per_segment, mac_lane, weight_cols_total):
    weights_col_per_segment_list = [0] * seg_num
    weights_col_per_segment_max = 0
    
    for i in range(seg_num):
        if ((i + 1) * weights_maclane_col_per_segment * mac_lane + (seg_num - i - 1) * (weights_maclane_col_per_segment - 1) * mac_lane) >= weight_cols_total:
            for ii in range(i + 1):
                weights_col_per_segment_list[ii] = weights_maclane_col_per_segment * mac_lane
                weights_col_per_segment_max = weights_maclane_col_per_segment * mac_lane
            for ii in range(i + 1, seg_num):
                weights_col_per_segment_list[ii] = (weights_maclane_col_per_segment - 1) * mac_lane
            break
    return (weights_col_per_segment_list, weights_col_per_segment_max)
    
def stage_latency(die_num, core_num, weight_cols, weight_rows, mac_lane, mac_num, psum_num, sram2_height, IA_rows, debug_on=False):
    """ 
    psum_num: how many mac_num cols are there in one calculation block
    weight_loading_latency: for a mac_numB data
    cal_latency: for a mac_num dot production
    psum_add_latency: for adding a mac_lane * mac_lane * 24bit of data
    psum_add_latency2: adding all mac_lane * mac_lane blocks in a die and read a mac_lane * mac_lane block from last sub-round and 
                       add the two mac_lane * mac_lane blocks and write the result amc_lane * mac_lane block into HBM3
    psum_add_latency3: adding all mac_lane * mac_lane blocks in a die and write the result amc_lane * mac_lane block into HBM3
    """
    
    latencys = latency()
    
    """ calculate the maximum weight columns assigned to die """
    weights_col_per_die = math.ceil(weight_cols / die_num) 
    weights_maclane_col_per_die = math.ceil(weights_col_per_die / mac_lane)
    weights_col_per_die_list = [0] * die_num
    weights_col_per_die_max = 0
    (weights_col_per_die_list, weights_col_per_die_max) = calculate_weights_col_assignment(die_num, weights_maclane_col_per_die, mac_lane, weight_cols)
    
    if debug_on:
        print(f"FC stage, weights_col_per_die_list: {weights_col_per_die_list}")

    """ calcualte how many rounds of calculation should a die goes through for weight """
    # Die with the maximum amount of weights dominates the latency
    # number of rounds to complete using weigths in col-dim
    rounds_weight = math.ceil(weights_col_per_die_max / sram2_height)
    weights_col_per_round = math.ceil(weights_col_per_die_max / rounds_weight)
    weights_maclane_col_per_round = math.ceil(weights_col_per_round / mac_lane)
    weights_col_per_round_list = [0] * rounds_weight
    weights_col_per_round_max = 0
    (weights_col_per_round_list, weights_col_per_round_max) = calculate_weights_col_assignment(rounds_weight, weights_maclane_col_per_round, mac_lane, weights_col_per_die_max)
    
    if debug_on:
        print(f"FC stage, weights_col_per_round_list: {weights_col_per_round_list}")    
        
    """ calcualte how many rounds of calculation should a die goes through for IA """
    # number of rounds to complete using IA in row-dim
    rounds_IA = math.ceil(IA_rows / sram2_height)
    IA_rows_per_round = math.ceil(IA_rows / rounds_IA)
    IA_maclane_row_per_round = math.ceil(IA_rows_per_round / mac_lane)
    IA_rows_per_round_list = [0] * rounds_IA
    IA_rows_per_round_max = 0
    (IA_rows_per_round_list, IA_rows_per_round_max) = calculate_weights_col_assignment(rounds_IA, IA_maclane_row_per_round, mac_lane, IA_rows)
    
    if debug_on:
        print(f"FC stage, IA_rows_per_round_list: {IA_rows_per_round_list}")    
    
    # FIXME Dummy implementation. We should consider corner cases under special circumstances
    sub_rounds = math.ceil(math.ceil(weight_rows / (mac_num * psum_num)) / core_num)

    if debug_on: 
        print(f"FC stage, rounds_weight: {rounds_weight}")
        print(f"FC stage, sub-rounds: {sub_rounds}")
        print(f"FC stage, rounds_IA: {rounds_IA}")
    
    for i in range(rounds_weight):
        for j in range(sub_rounds):
            # load weights
            # FIXME Here assume weight laoding latency greater than IA broadcast latency. Under certain cases this is not true
            latencys.weight_loading += weights_col_per_round_list[i]
            
            for k in range(rounds_IA):
                # calculate
                # number of mac-lane col in a calculation block
                maclane_col_num = weights_col_per_round_list[i] / mac_lane
                # number of mac-lane row in a calculation block
                maclane_row_num = IA_rows_per_round_list[k] / mac_lane
                
                if debug_on:
                    print(f"FC stage, in a calculation block, [maclane_col_num, maclane_row_num]: [{maclane_col_num}, {maclane_row_num}]")
                
                latencys.cal += maclane_col_num * maclane_row_num * mac_lane
                if j > 0:
                    # if this is not the first sub-round, we should get partial sum from previous sub-rounds and add to partial sum of this sub-round
                    latencys.psum_cube_loading += maclane_col_num * maclane_row_num
                # add (N+1) cubes together
                latencys.psum_cube_cal_N += maclane_col_num * maclane_row_num
                # write mac-lane * mac-lane block back into HMC
                latencys.psum_cube_wb += maclane_col_num * maclane_row_num 
        
    return latencys

def vector_matrix_mul_latency(weight_rows, weight_cols, mac_lane, mac_num, psum_num, rounds, mode, debug_on=False):
    """ 
    mode: 0 for K, 1 for V
    """
    latencys = latency()
    
    # weight loading of K/V
    latencys.weight_loading += math.ceil(weight_rows / mac_num) * weight_cols * rounds
    # calculation
    latencys.cal += math.ceil(weight_cols / mac_lane) * psum_num * rounds
    # partial sum addition within a vec-mat multiplication
    latencys.psum_vec_cal += math.ceil(weight_cols / mac_lane) * (psum_num - 1) * rounds
    if mode == 0:
        latencys.vec_wb_sram += rounds * math.ceil(weight_cols / mac_lane)
    else:
        latencys.vec_wb += rounds * math.ceil(weight_cols / mac_lane)
        
    if debug_on:
        print("in vector_matrix_mul_latency")
        print(f"latency.weight_loading: {math.ceil(weight_rows / mac_num) * weight_cols * rounds}")
        print(f"latency.cal: {math.ceil(weight_cols / mac_lane) * psum_num * rounds}")
        print(f"latency.psum_vec_cal: {math.ceil(weight_cols / mac_lane) * (psum_num - 1) * rounds}")
        if mode == 0:
            print(f"latency.vec_wb_sram: {rounds * math.ceil(weight_cols / mac_lane)}")
        else:
            print(f"latency.vec_wb: {rounds * math.ceil(weight_cols / mac_lane)}")
        
    return latencys
        
def matrix_matrix_mul_latency(IA_rows, weight_rows, weight_cols, mac_lane, mac_num, psum_num, rounds, mode, wb_mode=False, debug_on=False):
    """ 
    mode: 0 for K, 1 for V
    wb_mode: True for write back to SRAM, False for write back to HMC
    """
    latencys = latency()
    # number of mac-lane col in a calculation block
    maclane_col_num = math.ceil(weight_cols / mac_lane)
    # number of mac-lane row in a calculation block
    maclane_row_num = math.ceil(IA_rows / mac_lane)
    
    if debug_on:
        print(f"in matrix_matrix_mul_latency(), [maclane_col_num, maclane_row_num]: [{maclane_col_num}, {maclane_row_num}]")
    
    # weight loading of K/V
    # FIXME Should consider if weight loading is greater than IA loading
    latencys.weight_loading += math.ceil(weight_rows / mac_num) * weight_cols * rounds
    # calculation
    latencys.cal += maclane_col_num * maclane_row_num * psum_num * mac_lane * rounds
    # partial sum addition within a mat-mat-multiplication
    latencys.psum_vec_cal += maclane_col_num * maclane_row_num * (psum_num - 1) * mac_lane * rounds
    if (mode == 0) and wb_mode:
        latencys.vec_wb_sram += rounds * mac_lane * maclane_col_num * maclane_row_num
    else:
        latencys.vec_wb += rounds * mac_lane * maclane_col_num * maclane_row_num
    
    if debug_on:
        print("in matrix_matrix_mul_latency")
        print(f"latency.weight_loading: {math.ceil(weight_rows / mac_num) * weight_cols * rounds}")
        print(f"latency.cal: {maclane_col_num * maclane_row_num * psum_num * mac_lane * rounds}")
        print(f"latency.psum_vec_cal: {maclane_col_num * maclane_row_num * (psum_num - 1) * mac_lane * rounds}")
        if mode == 0:
            print(f"latency.vec_wb_sram: {rounds * mac_lane * maclane_col_num * maclane_row_num}")
        else:
            print(f"latency.vec_wb: {rounds * mac_lane * maclane_col_num * maclane_row_num}")
    
    return latencys

# ref: https://blog.csdn.net/kl28978113/article/details/108483918
def find_average_method2(number_list, arr_num):
    avg_arrays = []
    if len(number_list) == 0 or len(number_list) < arr_num:
        return avg_arrays
    
    sum_value = 0
    mean_value = 0
    number_list_float = []
    for num in number_list: 
        number_list_float.append(float(num))
        sum_value += float(num)
    
    mean_value = sum_value / float(arr_num)
 
    number_list_float.sort(reverse=True)
    
    for cnt in range(0, arr_num):
        arr = []
        if cnt == arr_num-1: 
            avg_arrays.append(transFloatToIntList(number_list_float))
            break
        
        if len(number_list_float) > 0 and number_list_float[0] >= mean_value: 
            arr = [number_list_float[0]]
            avg_arrays.append(transFloatToIntList(arr))
            sum_value = sum_value - number_list_float[0]
 
            mean_value = sum_value / float(arr_num-len(avg_arrays))
        else: 
            arr, _ = getList(number_list_float, mean_value, math.pow(mean_value, 2))
            avg_arrays.append(transFloatToIntList(arr))
        
        number_list_float = removeFromFloatList(number_list_float, arr)
 
    return avg_arrays
 
def transFloatToIntList(float_list):
    res = []
    for item in float_list:
        res.append(int(item))
    
    return res
 
def removeFromFloatList(original_list, remove_nums):
    res = []
    start = 0
    for remove in remove_nums:
        for i in range(start, len(original_list)):
            if original_list[i] == remove:
                res.extend(original_list[start:i])
                start = i + 1
                break
    
    res.extend(original_list[start:])
    return res
 
def getList(arr, delta, distance):
    res = []
    if len(arr) == 0:
        return res, -1
    
 
    for i in range(0, len(arr)-1):
        if delta == arr[i]:
            res.append(arr[i])
            return res, 0
        elif delta < arr[i]:
            continue
        elif delta > arr[i]:
            if i == 0:
                res.append(arr[i])
                delta = delta - arr[i]
                distance = math.pow(delta, 2)
                tmp, d = getList(arr[i+1:], delta, distance)
                res.extend(tmp)
                return res, d
            else:
                dis1 = math.pow(arr[i-1]-delta, 2)
                dis2 = math.pow(delta-arr[i], 2)
                if dis1 > dis2:
                    res.append(arr[i])
                    delta = delta - arr[i]
                    tmp, d = getList(arr[i+1:], delta, dis2)
                    res.extend(tmp)
                    return res, d
                else:
                    tmp, d = getList(arr[i:], delta, dis2)
                    if dis1 > d:
                        res.extend(tmp)
                        return res, d
                    
 
                    res.append(arr[i-1])
                    return res, dis1
       
    dis = math.pow(delta-arr[len(arr)-1], 2)
 
    if dis < distance:
        return arr[len(arr)-1:], dis
    
    return [], -1  
# endref

def find_average_method1(seq_len_table, group_num, die_idx_table):
    seq_len_table_copy = copy.deepcopy(seq_len_table)
    for i in range(len(seq_len_table_copy)):
        seq_len_table_copy[i] += (i,)
        
    ordered_seq_len_table = sorted(seq_len_table_copy, key=lambda x:(x[2]), reverse=True)
    
    for i in ordered_seq_len_table:
        print(i)
        
    avg_arr = []
    sum_arr = [] 
    for i in range(group_num):
        avg_arr.append([])
        sum_arr.append(0)
    
    for i in range(len(ordered_seq_len_table)):
        v = ordered_seq_len_table[i][2]
        for j in range(len(avg_arr)):
            sum_arr[j] = sum(avg_arr[j])
        idx = sum_arr.index(min(sum_arr))
        avg_arr[idx].extend([v])
        die_idx_table[ordered_seq_len_table[i][4]] = idx

def find_average_with_foundation_method1(new_seq_len_table, KV_buffering_amount_list, die_idx_table, new_seq_len_table_id):
    new_seq_len_table_copy = copy.deepcopy(new_seq_len_table)
    
    for i in range(len(new_seq_len_table_copy)):
        new_seq_len_table_copy[i] += (new_seq_len_table_id[i],)
        
    ordered_new_seq_len_table = sorted(new_seq_len_table_copy, key=lambda x:(x[2]), reverse=True)
    
    for i in ordered_new_seq_len_table:
        print(i)
    
    avg_arr = []
    sum_arr = []
    for i in range(len(KV_buffering_amount_list)):
        avg_arr.append([])
        sum_arr.append(0)
        
    for i in range(len(ordered_new_seq_len_table)):
        v = ordered_new_seq_len_table[i][2]
        for j in range(len(KV_buffering_amount_list)):
            sum_arr[j] = sum(avg_arr[j]) + KV_buffering_amount_list[j]
        idx = sum_arr.index(min(sum_arr))
        avg_arr[idx].extend([v])
        KV_buffering_amount_list[idx] += v
        die_idx_table[ordered_new_seq_len_table[i][4]] = idx
        
    
def MH_latency(die_num, core_num, seq_len_table, die_idx_table, B, H, sram2_height, mac_lane, mac_num, head_embedding_dim, debug_on=False):
    """ 
    seq_len_table: (if this time this job is in prefill stage, requests_id, sequence length, done?)
    """
    # FIXME since there's padding overhead for MH, when assign K/V to differnent dies, we may consider s after padding instead of raw s
    
    seg_latencys = latency()
    latencys = [latency()] * die_num
    KV_buffering_amount_list = []
    for i in range(die_num):
        KV_buffering_amount_list.append(0)
        
    """ assign new Q-As' K/V location """
    for i in range(B):
        # record each die's K/V buffering status
        if seq_len_table[i][0] == False:
            # if in generation stage
            KV_buffering_amount_list[die_idx_table[i]] += seq_len_table[i][2]
            
    # print("------ new round -------")
    # print(f"KV_buffering_amount_list before: {KV_buffering_amount_list}")
    # print(die_idx_table)
    # print(seq_len_table)
        
    new_seq_len_table = [x for i, x in enumerate(seq_len_table) if (seq_len_table[i][0] == True)]
    new_seq_len_table_id = [i for i, x in enumerate(seq_len_table) if (seq_len_table[i][0] == True)]
    
    if debug_on:
        print(f"MH stage, new_seq_len_table: {new_seq_len_table}")
        print(f"KV_buffering_amount_list before: {KV_buffering_amount_list}")
        
    # if at the very beginning  
    if len(new_seq_len_table) == B:
        find_average_method1(seq_len_table, die_num, die_idx_table)
        # if debug_on:
        #     for i in seq_len_table:
        #         print(i)
    else:
        # TODO Implement smarter algorithm
        # for new_jobs in new_seq_len_table:
        #     die_idx = KV_buffering_amount_list.index(min(KV_buffering_amount_list))
        #     KV_buffering_amount_list[die_idx] += new_jobs[2]
        #     die_idx_table[new_seq_len_table_id[new_jobs_idx]] = die_idx
        #     new_jobs_idx += 1
        find_average_with_foundation_method1(new_seq_len_table, KV_buffering_amount_list, die_idx_table, new_seq_len_table_id)
        
    if debug_on:
        print(f"KV_buffering_amount_list after: {KV_buffering_amount_list}")
        print("---------- die_idx_table -----------")
        for d in die_idx_table:
            print(d)
        
    for i in range(B):
            
        # this job assigned to which die    
        die_idx = die_idx_table[i]
        
        if debug_on:
            print(f"----------------- token {i} - die {die_idx} ------------------")
        
        # this job's overall generated sequence length
        s = seq_len_table[i][2]
        
        # FIXME consider what if H cannot be divided by core_num
        rounds = math.ceil(H / core_num)

        if seq_len_table[i][0]:
            if debug_on:
                print(f"------ prefill --------")
            # if prefill
            # FIXME  Should consider the case that SRAM can hold K/V but not twice the capacity of s*s matrix
            """ MH1 calculation """
            if sram2_height * mac_num < head_embedding_dim * s:
                # if SRAM cannot hold K/V
                seg_num = math.ceil(s / (sram2_height / math.ceil(head_embedding_dim / mac_num)))
                weights_col_per_seg = math.ceil(s / seg_num)
                weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, s)
                if debug_on:
                    print(f"MH1, SRAM cannot hold K/V")
                    print(f"MH1, weights_col_per_sram_list: {weights_col_per_sram_list}")
                
                psum_num = math.ceil(head_embedding_dim / mac_num)
                for weights_col in weights_col_per_sram_list:
                    # FIXME
                    for IA_rows in weights_col_per_sram_list:
                        seg_latencys = matrix_matrix_mul_latency(IA_rows, head_embedding_dim, weights_col, mac_lane, mac_num, psum_num, rounds, 0, debug_on)
                
            else:
                # if SRAM can hold K/V
                # FIXME Here we assume head_embedding_dim can be fully divided by mac_num
                if debug_on:
                    print(f"MH1, SRAM can hold K/V")
                    
                psum_num = math.ceil(head_embedding_dim / mac_num)
                seg_latencys = matrix_matrix_mul_latency(s, head_embedding_dim, s, mac_lane, mac_num, psum_num, rounds, 0, (sram2_height * mac_num >= 2 * head_embedding_dim * s), debug_on)
            latencys[die_idx] = latency_acc2(seg_latencys, latencys[die_idx])

            
            """ MH2 calculation """
            if sram2_height * mac_num < head_embedding_dim * s:
                # FIXME consider the case cannot hold A
                # if SRAM cannot hold K/V
                # FIXME here we assume s is less then 4096
                seg_num = math.ceil(head_embedding_dim / math.floor(sram2_height / math.ceil(s / mac_num)))    
                weights_col_per_seg = math.ceil(head_embedding_dim / seg_num)
                weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, head_embedding_dim)  
                if debug_on:
                    print(f"MH2, SRAM cannot hold K/V")
                    print(f"MH2, weights_col_per_sram_list: {weights_col_per_sram_list}")
                    
                # if debug_flag:
                #     print(f"MH2 weights_col_list: {weights_col_per_sram_list}")
                psum_num = math.ceil(s / mac_num)
                for weights_col in weights_col_per_sram_list:
                    for IA_rows in weights_col_per_sram_list:
                    # FIXME
                        seg_latencys = matrix_matrix_mul_latency(IA_rows, s, weights_col, mac_lane, mac_num, psum_num, rounds, 1, debug_on)    
                
            else:
                # if SRAM can hold K/V
                if debug_on:
                    print(f"MH2, SRAM can hold K/V")
                    
                psum_num = math.ceil(s / mac_num)
                seg_latencys = matrix_matrix_mul_latency(s, s, head_embedding_dim, mac_lane, mac_num, psum_num, rounds, 1, True, debug_on)
                
            latencys[die_idx] = latency_acc2(seg_latencys, latencys[die_idx])
            
        else: 
            if debug_on:
                print(f"-------- generation ----------")
                
            # if generation
            if sram2_height * mac_num < head_embedding_dim * s:
                # SRAM cannot hold K/V
                if debug_on:
                    print("sram cannot hold K/V")
                    
                # if SRAM2 cannot hold entire K at the same time
                """ MH1 calculation """
                seg_num = math.ceil(s / (sram2_height / math.ceil(head_embedding_dim / mac_num)))
                weights_col_per_seg = math.ceil(s / seg_num)
                weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, s)
                
                if debug_on:
                    print(f"MH1 weights_col_per_sram_list: {weights_col_per_sram_list}")
                    
                psum_num = math.ceil(head_embedding_dim / mac_num)
                for weights_col in weights_col_per_sram_list:
                    # FIXME
                    seg_latencys = vector_matrix_mul_latency(head_embedding_dim, weights_col, mac_lane, mac_num, psum_num, rounds, 0, debug_on)
                    
                latencys[die_idx] = latency_acc2(seg_latencys, latencys[die_idx])
                    
                """ MH2 calcualtion """   
                # FIXME here we assume s is less then 4096
                seg_num = math.ceil(head_embedding_dim / math.floor(sram2_height / math.ceil(s / mac_num)))    
                weights_col_per_seg = math.ceil(head_embedding_dim / seg_num)
                weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, head_embedding_dim)  
                
                if debug_on:
                    print(f"MH2 weights_col_per_sram_list: {weights_col_per_sram_list}")
                    
                psum_num = math.ceil(s / mac_num)
                for weights_col in weights_col_per_sram_list:
                    # FIXME
                    seg_latencys = vector_matrix_mul_latency(s, weights_col, mac_lane, mac_num, psum_num, rounds, 1, debug_on)    
                    
                latencys[die_idx] = latency_acc2(seg_latencys, latencys[die_idx])
            else:
                # SRAM can hold K/V
                if debug_on:
                    print("sram can hold K/V")
                    
                """ MH1 calculation """
                # FIXME Here we assume head_embedding_dim can be fully divided by mac_num
                psum_num = math.ceil(head_embedding_dim / mac_num)
                seg_latencys = vector_matrix_mul_latency(head_embedding_dim, s, mac_lane, mac_num, psum_num, rounds, 0, debug_on)
                latencys[die_idx] = latency_acc2(seg_latencys, latencys[die_idx])
                
                """ MH2 calculation """
                psum_num = math.ceil(s / mac_num)
                seg_latencys = vector_matrix_mul_latency(s, head_embedding_dim, mac_lane, mac_num, psum_num, rounds, 1, debug_on)
                latencys[die_idx] = latency_acc2(seg_latencys, latencys[die_idx])
        
    # print(latencys)
    # FIXME
    if debug_on:
        for l in latencys:
            l.dump()
        
    return latencys[0]

         
def simulation(args, requests_pool, Statistics):
    """ 
    requests_pool: (used, prompt_len, answer len)
    """
    
    """ HW """
    die_num = args.die_num
    mac_lane = args.mac_lane
    mac_num = args.mac_num
    sram2_height = args.SRAM2_height
    N = args.core_num_in_die
    
 
    """ SW """
    B = args.batch_size
    B = 8
    H = args.head_num
    D = args.decoder_num
    embedding_dim = args.embedding_dim
    head_embedding_dim = embedding_dim // H
        
    # how many requests are there in the requests pool - for simulation purpose
    requests_total_count = len(requests_pool)
    
    # record each Q-A's present sequence length
    # (if this time this job is in prefill stage, requests_id, seqeunce length, done?)
    # done is only for the last requets marking whether they completes
    seq_len_table = [(True, 0, 0, True)] * B
    
    # record each Q-A's assigned die idx
    die_idx_table = [0] * B
    
    """ seq_len_table initiation """
    for i in range(B):
        seq_len_table[i] = [True, i, requests_pool[i][1], False]
        requests_pool[i][0] = True
    # record which job will be feed into the system next time
    requests_id = B
    
    # rows of input activation height
    IA_rows = 0
    
    seg_latency = latency()
    total_latency = 0
    
    """ LATENCY """
    cal_latency = args.cal_latency
    psum_vec_cal_latency = args.psum_vec_cal_latency
    psum_cube_cal_N_latency = args.psum_cube_cal_N_latency
    psum_cube_loading_latency = args.psum_cube_loading_latency
    weight_loading_latency = args.weight_loading_latency
    psum_cube_wb_latency = args.psum_cube_wb_latency
    
    """ debug """
    debug_on = True
    
    print("requests_pool")
    for e in requests_pool:
        print(e)
    print(f"requests_id:{requests_id}")
    print(f"request_total_count:{requests_total_count}")
    
    if args.sim_mode == 0:
        while requests_done(seq_len_table) == False:
            if debug_on:
                print("--------------- new round ----------------")
                for e in seq_len_table:
                    print(e)
                
                
            """ profiling """
            IA_rows = 0
            for e in seq_len_table:
                if e[0]:
                    # if in prefill stage
                    IA_rows += e[2]
                else:
                    # in generation stage
                    IA_rows += 1
                    
            if debug_on:
                print(f"IA rows: {IA_rows}")
                        
            """ stage1 """
            # if r > 0:
            #     # FIXME Detail this 
            #     latency += IA_latency
            # latency += (H - 1) * IA_latency # FIXME
            if debug_on:
                print("------------- FC1 -------------")
            seg_latency = stage_latency(die_num, N, 3 * embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            if debug_on:
                seg_latency.dump()
                Statistics.dump()
                    
            """ MH """
            # latency += H * IA_latency # FIXME
            if debug_on:
                print("------------- MH -------------")
            seg_latency = MH_latency(die_num, N, seq_len_table, die_idx_table, B, H, sram2_height, mac_lane, mac_num, head_embedding_dim, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            if debug_on:
                seg_latency.dump()
                Statistics.dump()
            
            """ stage2 """
            # latency += H * IA_latency # FIXME
            if debug_on:
                print("------------- FC2 -------------")
            seg_latency = stage_latency(die_num, N, embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            if debug_on:
                seg_latency.dump()
                Statistics.dump()
            
            """ stage3 """
            # latency += H * IA_latency # FIXME
            if debug_on:
                print("------------- FC3 -------------")
            seg_latency = stage_latency(die_num, N, 4 * embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            if debug_on:
                seg_latency.dump()
                Statistics.dump()
            
            """ stage4 """
            # latency += H * IA_latency # FIXME
            if debug_on:
                print("------------- FC4 -------------")
            seg_latency = stage_latency(die_num, N, embedding_dim, 4 * embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            if debug_on:
                seg_latency.dump()
                Statistics.dump()
            
            debug_on = False
            """ at the end of the round, check if certain Q-A stops """
            for i in range(B):
                if seq_len_table[i][0] == False:
                    # in generation stage
                    # answer length ++
                    if seq_len_table[i][3] == False:
                        seq_len_table[i][2] += 1
                    if seq_len_table[i][2] == (requests_pool[seq_len_table[i][1]][1] + requests_pool[seq_len_table[i][1]][2]):
                        # if this job's generation completes, feed a new job into batch
                        if requests_id == requests_total_count:
                            # requests pool is run out
                            seq_len_table[i][3] = True
                        else:
                            assert (requests_pool[requests_id][0] == False)   # this must be an unused request 
                            seq_len_table[i] = [True, requests_id, requests_pool[requests_id][1], False]
                            requests_pool[requests_id][0] = True
                            requests_id += 1         
                            debug_on = True  
                else:
                    # in prefill stage, next time switch to generation stage
                    seq_len_table[i][0] = False
    elif args.sim_mode == 1:
        # fin the final small batch
        B_list = []
        group_num = requests_total_count // B
        for i in range(group_num):
            B_list.append(B)
        B_list.append(requests_total_count - group_num * B)
        print(f"B_list: {B_list}")
        
        # record which batch is now processing
        B_idx = 0
        for b in B_list:
            # find the max answer/prompt length to pad with
            requests_pool_seg = requests_pool[B_idx * B: (B_idx + 1) * B if b == B else requests_total_count]
            max_prompt_len = max(requests_pool_seg, key=itemgetter(1))[1] 
            max_answer_len = max(requests_pool_seg, key=itemgetter(2))[2] 
            print(f"[max_prompt_len, max_answer_len]: [{max_prompt_len}, {max_answer_len}]")
            for i in range(b):
                seq_len_table[i] = [True, i, max_prompt_len, False]
            
            while seq_len_table[0][2] < (max_prompt_len + max_answer_len):
                if seq_len_table[0][0] or (seq_len_table[0][2] == (max_prompt_len + 1)):
                    debug_on = True
                    
                if debug_on:
                    print("--------------- new round ----------------")
                    for e in seq_len_table:
                        print(e)
                
                """ profiling """
                IA_rows = 0
                for e in seq_len_table:
                    if e[0]:
                        # if in prefill stage
                        IA_rows += e[2]
                    else:
                        # in generation stage
                        IA_rows += 1
                        
                if debug_on:
                    print(f"IA rows: {IA_rows}")
                            
                """ stage1 """
                # if r > 0:
                #     # FIXME Detail this 
                #     latency += IA_latency
                # latency += (H - 1) * IA_latency # FIXME
                if debug_on:
                    print("------------- FC1 -------------")
                seg_latency = stage_latency(die_num, N, 3 * embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
                latency_acc1(seg_latency, Statistics, D)
                if debug_on:
                    seg_latency.dump()
                    Statistics.dump()
                        
                """ MH """
                # latency += H * IA_latency # FIXME
                if debug_on:
                    print("------------- MH -------------")
                seg_latency = MH_latency(die_num, N, seq_len_table, die_idx_table, b, H, sram2_height, mac_lane, mac_num, head_embedding_dim, debug_on)
                latency_acc1(seg_latency, Statistics, D)
                if debug_on:
                    seg_latency.dump()
                    Statistics.dump()
                
                """ stage2 """
                # latency += H * IA_latency # FIXME
                if debug_on:
                    print("------------- FC2 -------------")
                seg_latency = stage_latency(die_num, N, embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
                latency_acc1(seg_latency, Statistics, D)
                if debug_on:
                    seg_latency.dump()
                    Statistics.dump()
                
                """ stage3 """
                # latency += H * IA_latency # FIXME
                if debug_on:
                    print("------------- FC3 -------------")
                seg_latency = stage_latency(die_num, N, 4 * embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
                latency_acc1(seg_latency, Statistics, D)
                if debug_on:
                    seg_latency.dump()
                    Statistics.dump()
                
                """ stage4 """
                # latency += H * IA_latency # FIXME
                if debug_on:
                    print("------------- FC4 -------------")
                seg_latency = stage_latency(die_num, N, embedding_dim, 4 * embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
                latency_acc1(seg_latency, Statistics, D)
                if debug_on:
                    seg_latency.dump()
                    Statistics.dump()
                
                for e in seq_len_table:
                    e[2] += 1
                    if e[0]:
                        e[0] = False
                        
                debug_on = False
        
            B_idx += 1
                   
    total_latency += Statistics.latency.cal * cal_latency
    total_latency += Statistics.latency.psum_vec_cal * psum_vec_cal_latency
    total_latency += Statistics.latency.psum_cube_cal_N * psum_cube_cal_N_latency
    total_latency += Statistics.latency.psum_cube_loading * psum_cube_loading_latency
    total_latency += Statistics.latency.weight_loading * weight_loading_latency 
    total_latency += Statistics.latency.psum_cube_wb * psum_cube_wb_latency
        
def main():
    """ Main function """

    args = argparser().parse_args()
    requests_pool = init_requests_pool()
    Statistics = Stats()
    simulation(args, requests_pool, Statistics)
        
    return 0
    
if __name__ == '__main__':
    main()