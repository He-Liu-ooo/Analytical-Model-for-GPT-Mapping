import argparse
import random
import math
import copy
from operator import itemgetter
from utils import Stats
from utils import latency

""" 
TODO
2. maybe there are better algorithm for prefill stage distribution and prefill & generation stage distribution, for now they all use the same simple algorithm
8. handle to overlapped of loading and writing IA between different stages(this problem is almost done, but not complete)
"""

""" 
NOTE
1. no FC-IA mode, since we choose weight partition, the result maxtrix(next layer's IA) is naturally partitioned in dies
2. no softmax mode, since we can do local maximum, data amount required for routing is the same as linear-approximate version
3. core-level and die-level distribution, the latency difference lies in whether concat is needed
4. the direct reuse between split and MH1 is hard to quantifiy, therefore we ignore this
"""
debug_f = False

def argparser():
    """ Argument parser. """

    ap = argparse.ArgumentParser()

    # HW
    ap.add_argument('--die-num', type = int, default = 4,
                    help = 'number of dies in the system')
    ap.add_argument('--core-num-in-die', type = int, default = 16,     
                    help = 'number of cores in a die')
    ap.add_argument('--mac-lane', type = int, default = 16)
    ap.add_argument('--mac-num', type = int, default = 32)
    ap.add_argument('--SRAM2_height', type = int, default = 2048,
                    help = 'number of 32B can be stored in a core\'s SRAM2')
    
    # MAPPING
    ap.add_argument('--W', type = int, default = 1,
                    help = 'parameter when calculating FC layer, for every core, determine the width of dot production, width = mac_num * W')
    # ap.add_argument('--softmax-mode', type = int, default = 0,
    #                 help = '0 for original softmax, 1 for optimized softmax')
    
    # MODEL
    ap.add_argument('--batch-size', type = int, default = 128,
                    help = 'number of tokens in a big batch')
    ap.add_argument('--embedding-dim', type = int, default = 4096,
                    help = 'embedding dimension')
    ap.add_argument('--head-num', type = int, default = 32,
                    help = 'number of heads in multihead attention')
    ap.add_argument('--decoder-num', type = int, default = 32,
                    help = 'number of decoders in the model')
    
    # ALGORITHM
    ap.add_argument('--P', type = float, default = 0.1,
                    help = 'percentage of job number that determines the maximum and minimum boundary of job number a die can hold')
    # TODO this option can be replaced by adjusting P to certain extreme values
    ap.add_argument('--MH-alg-distr-mode', type = int, default = 0,
                    help = '0 for origin distribution algorithm, 1 for distribution algorithm with boundary')
    ap.add_argument('--MH-alg-level-mode', type = int, default = 0,
                    help = '0 for implementation the load-balance algorithm in die level, 1 for core level')

    # PARAM
    # TODO this parameter scales with number of cores 
    ap.add_argument('--cal-latency', type = int, default = 64,
                    help = 'latency of calculating a mac_num dot production(8bit)')
    # ap.add_argument('--psum-vec-cal-latency', type = int, default = 0.5,
    #                 help = 'latency of adding 2 mac_lane * 3Byte data')
    # ap.add_argument('--psum-cube-cal-latency', type = int, default = 0.5,
    #                 help = 'latency of adding 2 mac_lane * mac_lane * 3Byte data')
    # ap.add_argument('--psum-cube-cal-N-latency', type = int, default = 0.5,
    #                 help = 'latency of adding N(core number in a die) mac_lane * mac_lane * 3Byte data')
    # ap.add_argument('--psum-cube-loading-latency', type = int, default = 0.5,
    #                 help = 'latency of reading a mac_lane * mac_lane * 3Byte data from HMC')
    ap.add_argument('--ppu-latency', type = int, default = 5,
                    help = 'latency of a PPU operation with data amount boundary')
    ap.add_argument('--ppu-boundary', type = int, default = 100,
                    help = 'PPU data amount boundary, in the unit of Byte')
    # PPU latency = math.ceil(data_amount/ppu_boundary)*ppu_latency
    
    # ap.add_argument('--add-8-latency', type = int, default = 0.5,
    #                 help = 'latency of adding 2 8-bit data together')
    # ap.add_argument('--add-24-latency', type = int, default = 0.5,
    #                 help = 'latency of adding 2 24-bit data together')
    # ap.add_argument('--substract-square-latency', type = int, default = 0.5,
    #                 help = 'latency of adding 2 (oprand1 - oprand2)^2, one oprand is 8-bit and the other is 24-bit')
    # ap.add_argument('--division-latency', type = int, default = 0.5,
    #                 help = 'latency of divide two data, one is 24-bit, one is 8-bit')
    # ap.add_argument('--LN-latency', type = int, default = 0.5,
    #                 help = 'latency of (x-EX)/(sqrt(Var+epsilon))*gamma + beta')
    # ap.add_argument('--find_max_latency', type = int, default = 0.5,
    #                 help = 'latency of finding maximum among mac_lane data')
    # ap.add_argument('--substract_exp_latency', type = int, default = 0.5,
    #                 help = 'latency of e^(x-x_max) of a data')
    ap.add_argument('--weight-loading-latency', type = int, default = 1,
                    help = 'latency of transfering mac_num Byte of data from HMC to SRAM of core')
    # ap.add_argument('--psum-cube-wb-latency', type = int, default = 0.5,
    #                 help = 'latency of writing a mac_lane * mac_lane * 3Byte data to HMC')
    # ap.add_argument('--csum-vec-wb-latency', type = int, default = 0.5,
    #                 help = 'latency of writing a complete mac_lane Byte data to HMC')
    # ap.add_argument('--csum-vec-wb-sram-latency', type = int, default = 0.5,
    #                 help = 'latency of writing a complete mac_lane Byte data to SRAM of core')
    
    ap.add_argument('--NoC-core-bandwidth', type = int, default = 68,
                    help = 'core bandwidth, in the unit of Byte/cycle')
    ap.add_argument('--NoP-die-bandwidth', type = int, default = 100,
                    help = 'die bandwidth, in the unit of Byte/cycle')
    # TODO: we need to check this
    ap.add_argument('--ppu-bandwidth', type = int, default = 120,
                    help = 'ppu bandwidth, in the unit of Byte/cycle')
    ap.add_argument('--NoC-latency-per-hop', type = int, default = 10)
    ap.add_argument('--NoP-latency-per-hop', type = int, default = 20)
    ap.add_argument('--NoC-h', type = int, default = 4,
                    help = 'number of cores within a die in y axis')
    ap.add_argument('--NoC-w', type = int, default = 4,
                    help = 'number of cores within a die in x axis')
    ap.add_argument('--NoP-h', type = int, default = 2,
                    help = 'number of dies in y axis')
    ap.add_argument('--NoP-w', type = int, default = 2,
                    help = 'number of dies in x axis')
    ap.add_argument('--is-ring-mode', type = int, default = True,
                    help = 'whether die is of ring topology')
    # NoC latency of X Byte data = math.ceil(X/NoC-core-bandwidth) + NoC-latency-per-hop*hop_num
    # NoP latency of X Byte data = math.ceil(X/NoP-core-bandwidth) + NoP-latency-per-hop*hop_num
    
    # OTHERS
    ap.add_argument('--debug-flag', type = bool, default = True)       
    ap.add_argument('--sim-mode', type = int, default = 0,
                    help = '0 for my design, 1 for FlexGen prompt/answer padding implementation')
    
    return ap

def dump_res(total_latency, fc_latency, mh_latency, nonlinear_latency, nonlinear_latencys, FC_Statistics, MH_Statistics, NonLinear_Statistics, p_NonLinear_Statistics, Statistics):
    print("================================ Results ===============================")
    Statistics.dump("Total")
    Statistics.latency.dump_portion("Total")
    FC_Statistics.dump("FC")
    FC_Statistics.latency.dump_portion("FC")
    MH_Statistics.dump("MH")
    MH_Statistics.latency.dump_portion("MH")
    NonLinear_Statistics.dump("NonLinear")
    NonLinear_Statistics.latency.dump_portion("NonLinear")
    
    p_NonLinear_Statistics[0].dump("Layer Norm")
    p_NonLinear_Statistics[0].latency.dump_portion("Layer Norm")
    p_NonLinear_Statistics[1].dump("Split")
    p_NonLinear_Statistics[1].latency.dump_portion("Split")
    p_NonLinear_Statistics[2].dump("Softmax")
    p_NonLinear_Statistics[2].latency.dump_portion("Softmax")
    p_NonLinear_Statistics[3].dump("Concat")
    p_NonLinear_Statistics[3].latency.dump_portion("Concat")
    p_NonLinear_Statistics[4].dump("Residual")
    p_NonLinear_Statistics[4].latency.dump_portion("Residual")
    p_NonLinear_Statistics[5].dump("IA Rotation")
    p_NonLinear_Statistics[5].latency.dump_portion("IA Rotation")
    
    print(f"+ total latency: {total_latency}")
    print(f"+ fc latency: {fc_latency}")
    print(f"+ mh latency: {mh_latency}")
    print(f"+ nonlinear latency: {nonlinear_latency}")
    print(f"+ ln latency: {nonlinear_latencys[0]}")
    print(f"+ split latency: {nonlinear_latencys[1]}")
    print(f"+ softmax latency: {nonlinear_latencys[2]}")
    print(f"+ concat latency: {nonlinear_latencys[3]}")
    print(f"+ residual latency: {nonlinear_latencys[4]}")
    print(f"+ IA rotation latency: {nonlinear_latencys[5]}")

def init_requests_pool():
    requests_pool = []
    
    with open("GPT_requests_pool.txt", "r") as f:
        ss = f.readlines()
        for s in ss:
            s = s.strip('\n')
            s = s.split(',')
            requests_pool.append([False, int(s[0]), int(s[1])])
            
    s_sum = 0        
    for r in requests_pool:
        s_sum += (r[1] + r[2])
        
    return (requests_pool, s_sum / len(requests_pool))
        
def requests_done(seq_len_table):
    done = True
    for i in seq_len_table:
        if i[3] == False:
            done = False
            break
    return done

def latency_acc1(stage_latency, Statistics, D):
    Statistics.latency.cal += stage_latency.cal * D
    Statistics.latency.ppu += stage_latency.ppu * D
    Statistics.latency.weight_loading += stage_latency.weight_loading * D
    Statistics.latency.dynamic_weight_loading += stage_latency.dynamic_weight_loading * D
    Statistics.latency.IA_loading += stage_latency.IA_loading * D
    Statistics.latency.NoC_latency += stage_latency.NoC_latency * D
    Statistics.latency.NoP_latency += stage_latency.NoP_latency * D 
    
def latency_substract(FC_Statistics, nonlinear_latency):
    FC_Statistics.latency.cal -= nonlinear_latency.cal
    FC_Statistics.latency.ppu -= nonlinear_latency.ppu
    FC_Statistics.latency.weight_loading -= nonlinear_latency.weight_loading
    FC_Statistics.latency.dynamic_weight_loading -= nonlinear_latency.dynamic_weight_loading
    FC_Statistics.latency.IA_loading -= nonlinear_latency.IA_loading
    FC_Statistics.latency.NoC_latency -= nonlinear_latency.NoC_latency
    FC_Statistics.latency.NoP_latency -= nonlinear_latency.NoP_latency

def latency_acc2(stage_latency, latencys):
    new_latencys = latency()
    
    new_latencys.cal = latencys.cal + stage_latency.cal
    new_latencys.ppu = latencys.ppu + stage_latency.ppu
    new_latencys.weight_loading = latencys.weight_loading + stage_latency.weight_loading
    new_latencys.dynamic_weight_loading = latencys.dynamic_weight_loading + stage_latency.dynamic_weight_loading
    new_latencys.IA_loading = latencys.IA_loading + stage_latency.IA_loading
    new_latencys.NoC_latency = latencys.NoC_latency + stage_latency.NoC_latency
    new_latencys.NoP_latency = latencys.NoP_latency + stage_latency.NoP_latency
    
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
    
def stage_latency(die_num, core_num, weight_cols, weight_rows, mac_lane, mac_num, psum_num, sram2_height, IA_rows, p_NonLinear_Statistics, params, FC_no, is_LN_before=False, is_residual_after=False, debug_on=False):
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
    latencys.get_params(params)
    nonlinear_latencys = latency()
    nonlinear_latencys.get_params(params)
    
    if debug_f:
        print(f"---------- FC {FC_no} start ------------")
        print(f"is_LN_before: {is_LN_before}")
    
    """ calculate IA partition in case of IA rotation """
    # number of IA_rows * mac_num * psum_num sub-matrix of IA
    big_col_num = math.ceil(weight_rows / mac_num / psum_num)
    IA_big_col_per_die = math.ceil(big_col_num / die_num)
    IA_core_big_col_per_die = math.ceil(IA_big_col_per_die / core_num)
    # every element in the list can be divided by core number
    IA_big_col_per_die_list = [0] * die_num
    (IA_big_col_per_die_list, _) = calculate_weights_col_assignment(die_num, IA_core_big_col_per_die, core_num, big_col_num)
    # record 
    IA_rounds_per_die_list = []
    if debug_f:
        print(f"IA_big_col_per_die_list: {IA_big_col_per_die_list}")
    
    for e in IA_big_col_per_die_list:
        IA_rounds_per_die_list.append(math.ceil(e / core_num))
    if debug_f:
        print(f"IA_rounds_per_die_list: {IA_rounds_per_die_list}")        
    
    """ calculate the maximum weight columns assigned to die """
    weights_col_per_die = math.ceil(weight_cols / die_num) 
    weights_maclane_col_per_die = math.ceil(weights_col_per_die / mac_lane)
    weights_col_per_die_list = [0] * die_num
    weights_col_per_die_max = 0
    (weights_col_per_die_list, weights_col_per_die_max) = calculate_weights_col_assignment(die_num, weights_maclane_col_per_die, mac_lane, weight_cols)
    if debug_f: 
        print(f"weights_col_per_die_list: {weights_col_per_die_list}")
    
    # if debug_on:
    #     print(f"FC stage, weights_col_per_die_list: {weights_col_per_die_list}")

    """ calcualte how many rounds of calculation should a die goes through for weight """
    # Die with the maximum amount of weights dominates the latency
    # number of rounds to complete using weigths in col-dim
    rounds_weight = math.ceil(weights_col_per_die_max / (sram2_height / psum_num))
    weights_col_per_round = math.ceil(weights_col_per_die_max / rounds_weight)
    weights_maclane_col_per_round = math.ceil(weights_col_per_round / mac_lane)
    weights_col_per_round_list = [0] * rounds_weight
    weights_col_per_round_max = 0
    (weights_col_per_round_list, weights_col_per_round_max) = calculate_weights_col_assignment(rounds_weight, weights_maclane_col_per_round, mac_lane, weights_col_per_die_max)
    if debug_f:
        print(f"weights_col_per_round_list: {weights_col_per_round_list}")
    
    # if debug_on:
    #     print(f"FC stage, weights_col_per_round_list: {weights_col_per_round_list}")    
        
    """ calcualte how many rounds of calculation should a die goes through for IA """
    # number of rounds to complete using IA in row-dim
    rounds_IA = math.ceil(IA_rows / (sram2_height / psum_num))
    IA_rows_per_round = math.ceil(IA_rows / rounds_IA)
    IA_maclane_row_per_round = math.ceil(IA_rows_per_round / mac_lane)
    IA_rows_per_round_list = [0] * rounds_IA
    IA_rows_per_round_max = 0
    (IA_rows_per_round_list, IA_rows_per_round_max) = calculate_weights_col_assignment(rounds_IA, IA_maclane_row_per_round, mac_lane, IA_rows)
    if debug_f:
        print(f"IA_rows_per_round_list: {IA_rows_per_round_list}")
    
    # if debug_on:
    #     print(f"FC stage, IA_rows_per_round_list: {IA_rows_per_round_list}")    
    
    # FIXME Dummy implementation. We should consider corner cases under special circumstances
    sub_rounds = math.ceil(math.ceil(weight_rows / (mac_num * psum_num)) / core_num)              
    
    if debug_f:
        print(f"rounds_weight: {rounds_weight}, rounds_IA: {rounds_IA}, sub_rounds: {sub_rounds}")
    
    """ calculation with rotation """
    # record which partition of IA in the first die is now processing
    # NOTE: weight loading and IA loading cannot hide each other since they all come from the same CH
    partition_idx = 0
    tmp = 0
    calculation_print = True
    # operand
    IA_loading1 = 0
    # partial sum loading
    IA_loading2 = 0
    # total write back
    IA_loading3 = 0
    # rotation loading
    IA_loading4 = 0
    # rotation write back
    IA_loading5 = 0
    
    for i in IA_rounds_per_die_list:
        for j in range(rounds_weight):
            for k in range(i):
                
                # if weights_col_per_round_list[j] >= IA_rows_per_round_list[0]:
                    # weight loading latency hides IA loading latency
                    
                # every sub round means a new weight loading
                latencys.weight_loading += weights_col_per_round_list[j] * psum_num
                
                # else:
                    # IA loading latency hides weight loading latency
                    # if (partition_idx > 0) and (k == 0):
                        # since rotation will transfer data from other die onto the logic die, first subround of IA 
                        # latencys.weight_loading += weights_col_per_round_list[j] * psum_num
                    # else:
                        # latencys.IA_loading += IA_rows_per_round_list[0] * psum_num
                    
                for l in range(rounds_IA):
                    # laoding IA, but do not need to load weights
                    latencys.IA_loading += IA_rows_per_round_list[l] * psum_num
                    IA_loading1 += IA_rows_per_round_list[l] * psum_num
                    
                    # there are sevral cases IA loading is not needed
                    # 1. very first round, thanks to LN
                    if (l == 0) and (k == 0) and (j == 0) and (partition_idx == 0) and is_LN_before:
                        latencys.IA_loading -= IA_rows_per_round_list[l] * psum_num
                        IA_loading1 -= IA_rows_per_round_list[l] * psum_num
                    # 2. very first round of a partition, thanks to rotation
                    elif (partition_idx > 0) and (l == 0) and (k == 0) and (j == 0):
                        latencys.IA_loading -= IA_rows_per_round_list[l] * psum_num
                        IA_loading1 -= IA_rows_per_round_list[l] * psum_num
                    
                    # if ((k > 0) or (partition_idx == 0)):
                        
                    # calculate
                    # number of mac-lane col in a calculation block
                    maclane_col_num = math.ceil(weights_col_per_round_list[j] / mac_lane)
                    # number of mac-lane row in a calculation block
                    maclane_row_num = math.ceil(IA_rows_per_round_list[l] / mac_lane)
                    
                    latencys.cal += maclane_col_num * maclane_row_num * mac_lane * psum_num
                    # latencys.psum_cube_cal += (psum_num - 1) * maclane_col_num * maclane_row_num 
                    
                    if k > 0 or partition_idx > 0:
                        # if this is not the first sub-round, we should get partial sum from previous sub-rounds and add to partial sum of this sub-round
                        # NOTE: partial sum comes from all HMCs
                        latencys.IA_loading += math.ceil(3 * maclane_col_num * maclane_row_num * math.pow(mac_lane, 2) / mac_num / core_num)
                        IA_loading2 += math.ceil(3 * maclane_col_num * maclane_row_num * math.pow(mac_lane, 2) / mac_num / core_num)
                    if (partition_idx == (len(IA_rounds_per_die_list) - 1)) and (k == (i - 1)) and is_residual_after:
                        # if this is the last sub-round and there's a residual connection afterwards
                        latencys.IA_loading += math.ceil(maclane_col_num * maclane_row_num * math.pow(mac_lane, 2) / mac_num / core_num)
                        p_NonLinear_Statistics[4].latency.IA_loading += math.ceil(maclane_col_num * maclane_row_num * math.pow(mac_lane, 2) / mac_num / core_num)
                        nonlinear_latencys.IA_loading += math.ceil(maclane_col_num * maclane_row_num * math.pow(mac_lane, 2) / mac_num / core_num)
                        tmp += math.ceil(maclane_col_num * maclane_row_num * math.pow(mac_lane, 2) / mac_num / core_num)
                        
                    # add N/(N+1)/N+2 cubes together
                    # route to PPU
                    latencys.add_NoC_cores_to_ppu(3 * maclane_col_num * maclane_row_num * math.pow(mac_lane, 2))
                    # latencys.psum_cube_cal_N += maclane_col_num * maclane_row_num
                    # calculate
                    latencys.add_ppu_count(3 * maclane_col_num * maclane_row_num * math.pow(mac_lane, 2) * core_num)
                    # route to cores
                    # FIXME this is not precise, since data can be transferred to mutiple cores
                    # still partial sum here
                    latencys.add_NoC_ppu_to_core(3 * maclane_col_num * maclane_row_num * math.pow(mac_lane, 2))
                    # write mac-lane * mac-lane block back into HMC
                    # write back partial sum or complete sum
                    latencys.IA_loading += (1 if (partition_idx == (len(IA_rounds_per_die_list) - 1)) and (k == (i - 1)) else 3) * maclane_col_num * maclane_row_num * math.ceil(math.pow(mac_lane, 2) / mac_num / core_num)
                    IA_loading3 += (1 if (partition_idx == (len(IA_rounds_per_die_list) - 1)) and (k == (i - 1)) else 3) * maclane_col_num * maclane_row_num * math.ceil(math.pow(mac_lane, 2) / mac_num / core_num)

                    if calculation_print and debug_f:
                        print(f"(maclane_row_num, maclane_col_num): ({maclane_row_num}, {maclane_col_num})")
                        calculation_print = False
        # a partition of IA finishes calculating, we need to rotate partitions among dies
        # step1: read all the partition data from HMC to logic die
        # step2: transfer to other die's logic die
        # step3: write all the partition data from logic die to HMC
        # NOTE: first sub-round of a partition moves to calculation immediately without writing into HMC first     
        #       IA loading has a core_num parallelism
        #       we rotate IA, not results!
        # FIXME: here we use the first die's latency to represent the total latency, however, there exists case that the first die is not the die with the longest processing latency 
        if partition_idx + 1 < len(IA_rounds_per_die_list):          
            # last subround is already in SRAM, no loading
            latencys.IA_loading += max(math.ceil(IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows / core_num) - psum_num * IA_rows_per_round_list[-1], 0)
            p_NonLinear_Statistics[5].latency.IA_loading += max(math.ceil(IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows / core_num) - psum_num * IA_rows_per_round_list[-1], 0)
            nonlinear_latencys.IA_loading += max(math.ceil(IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows / core_num) - psum_num * IA_rows_per_round_list[-1], 0)
            IA_loading4 += max(math.ceil(IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows / core_num) - psum_num * IA_rows_per_round_list[-1], 0)
            
            # sending & receriving
            latencys.add_NoC_cores_to_ppu(IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows * mac_num / core_num)
            latencys.add_NoP_rotation(IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows * mac_num)  
            latencys.add_NoC_ppu_to_cores_m(IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows * mac_num / core_num)
            p_NonLinear_Statistics[5].latency.add_NoC_cores_to_ppu(IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows * mac_num / core_num)
            p_NonLinear_Statistics[5].latency.add_NoP_rotation(IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows * mac_num)  
            p_NonLinear_Statistics[5].latency.add_NoC_ppu_to_cores_m(IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows / core_num)
            nonlinear_latencys.add_NoC_cores_to_ppu(IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows * mac_num / core_num)
            nonlinear_latencys.add_NoP_rotation(IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows * mac_num)  
            nonlinear_latencys.add_NoC_ppu_to_cores_m(IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows / core_num)
            
            # the first sub-round's data don't need to store 
            latencys.IA_loading += max(math.ceil(IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows / core_num) - psum_num * IA_rows_per_round_list[0], 0)
            p_NonLinear_Statistics[5].latency.IA_loading += max(math.ceil(IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows / core_num) - psum_num * IA_rows_per_round_list[0], 0)
            nonlinear_latencys.IA_loading += max(math.ceil(IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows / core_num) - psum_num * IA_rows_per_round_list[0], 0)
            IA_loading5 += max(math.ceil(IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows / core_num) - psum_num * IA_rows_per_round_list[0], 0)
            
            if debug_on:
                print("---------------------- FC partition start --------------------")
                print(f"IA_rounds_per_die_list: {IA_rounds_per_die_list}")
                print(f"IA_big_col_per_die_list: {IA_big_col_per_die_list}")
                print(f"partition_idx: {partition_idx}")
                print(f"IA_loading: {IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows}")
                print(f"NoP_8: {IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows + IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows}")
                print(f"IA_writing: {max(IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows - core_num * psum_num * IA_rows, 0)}")
                print("---------------------- FC partition end --------------------")
                               
            partition_idx += 1 
            
    if debug_f: 
        print(f"IA_loading1: {IA_loading1}")
        print(f"IA_loading2: {IA_loading2}")
        print(f"IA_loading3: {IA_loading3}")
        print(f"IA_loading4: {IA_loading4}")
        print(f"IA_loading5: {IA_loading5}")
        print("------------- Residual Connections start -----------")             
        print(f"Residual IA loading: {tmp}")
        print("------------- Residual Connections end -------------")             
                
        print(f"---------- FC {FC_no} end ------------")
    # latencys.dump()                    
    return (latencys, nonlinear_latencys)

def vector_matrix_mul_latency(weight_rows, weight_cols, mac_lane, mac_num, psum_num, rounds, MH_mode, is_first=True, debug_on=False):
    """ 
    MH_mode: 0 for K, 1 for V
    is_first: for sram cannot hold K/V, if this is the first portion of calculation, we should load IA
    """
    latencys = latency()
    
    # weight loading of K/V
    latencys.dynamic_weight_loading += math.ceil(weight_rows / mac_num) * weight_cols * rounds
    # calculation
    latencys.cal += math.ceil(weight_cols / mac_lane) * psum_num * rounds
    # partial sum addition within a vec-mat multiplication
    # latencys.psum_vec_cal += math.ceil(weight_cols / mac_lane) * (psum_num - 1) * rounds
    if MH_mode == 0 and is_first:
        # MH1, we need load IA, and write result back to SRAM
        latencys.IA_loading += math.ceil(weight_rows / mac_num) * rounds
        # latencys.IA_loading += math.ceil(weight_rows / mac_num) * rounds
        # latencys.vec_wb_sram += rounds * math.ceil(weight_cols / mac_lane)
    else:
        # latencys.vec_wb += rounds * math.ceil(weight_cols / mac_lane)
        latencys.IA_loading += rounds * math.ceil(weight_cols / mac_num)
        
    if debug_on:
        print("in vector_matrix_mul_latency")
        print(f"latency.weight_loading: {math.ceil(weight_rows / mac_num) * weight_cols * rounds}")
        print(f"latency.cal: {math.ceil(weight_cols / mac_lane) * psum_num * rounds}")
        print(f"latency.psum_vec_cal: {math.ceil(weight_cols / mac_lane) * (psum_num - 1) * rounds}")
        if MH_mode == 0:
            print(f"latency.vec_wb_sram: {rounds * math.ceil(weight_cols / mac_lane)}")
        else:
            print(f"latency.vec_wb: {rounds * math.ceil(weight_cols / mac_lane)}")
        
    return latencys
        
def matrix_matrix_mul_latency(IA_rows, weight_rows, weight_cols, mac_lane, mac_num, psum_num, rounds, MH_mode, wb_mode=False, debug_on=False):
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
    latencys.dynamic_weight_loading += math.ceil(weight_rows / mac_num) * weight_cols * rounds
    # IA loading
    if MH_mode == 0:
        # MH1, we need to load IA
        latencys.IA_loading += math.ceil(weight_rows / mac_num) * rounds * IA_rows
    if (MH_mode == 1) and wb_mode:
        # MH2, and MH1 write back
        latencys.IA_loading += math.ceil(weight_rows / mac_num) * rounds * IA_rows
    # calculation
    latencys.cal += maclane_col_num * maclane_row_num * psum_num * mac_lane * rounds
    # partial sum addition within a mat-mat-multiplication
    # latencys.psum_vec_cal += maclane_col_num * maclane_row_num * (psum_num - 1) * mac_lane * rounds
    if (MH_mode == 0) and (wb_mode == False):
        # latencys.vec_wb_sram += rounds * mac_lane * maclane_col_num * maclane_row_num
        # MH1 and write back to SRAM
        pass
    else:
        # latencys.vec_wb += rounds * mac_lane * maclane_col_num * maclane_row_num
        # write back to HMC
        latencys.IA_loading += rounds * math.ceil(math.pow(mac_lane, 2) / mac_num) * maclane_col_num * maclane_row_num
        
    if debug_on:
        print("in matrix_matrix_mul_latency")
        print(f"latency.weight_loading: {math.ceil(weight_rows / mac_num) * weight_cols * rounds}")
        print(f"latency.cal: {maclane_col_num * maclane_row_num * psum_num * mac_lane * rounds}")
        print(f"latency.psum_vec_cal: {maclane_col_num * maclane_row_num * (psum_num - 1) * mac_lane * rounds}")
        if MH_mode == 0:
            print(f"latency.vec_wb_sram: {rounds * mac_lane * maclane_col_num * maclane_row_num}")
        else:
            print(f"latency.vec_wb: {rounds * mac_lane * maclane_col_num * maclane_row_num}")
    
    return latencys

def find_average_method1(seq_len_table, group_num, group_idx_table, J_max=0, J_min=0, MH_alg_distr_mode=0):
    # print(f"group_num: {group_num}")
    # print(f"MH_alg_distr_mode: {MH_alg_distr_mode}")
    seq_len_table_copy = copy.deepcopy(seq_len_table)
    for i in range(len(seq_len_table_copy)):
        seq_len_table_copy[i] += (i,)
        
    ordered_seq_len_table = sorted(seq_len_table_copy, key=lambda x:(x[2]), reverse=True)
    # for i in ordered_seq_len_table:
    #     print(i)
    
    # for i in ordered_seq_len_table:
    #     print(i)
        
    avg_arr = []
    sum_arr = [] 
    for i in range(group_num):
        avg_arr.append([])
        sum_arr.append(0)
    
    if MH_alg_distr_mode == 0:
        # origin distribution
        for i in range(len(ordered_seq_len_table)):
            v = ordered_seq_len_table[i][2]
            for j in range(len(avg_arr)):
                sum_arr[j] = sum(avg_arr[j])
            idx = sum_arr.index(min(sum_arr))
            avg_arr[idx].extend([v])
            group_idx_table[ordered_seq_len_table[i][5]] = idx
    else:
        # distribution with boundary
        group_job_table = []
        for i in range(group_num):
            group_job_table.append(0)
            
        for i in range(len(ordered_seq_len_table)):
            v = ordered_seq_len_table[i][2]
            for j in range(len(avg_arr)):
                sum_arr[j] = sum(avg_arr[j])
            idx = sum_arr.index(min(sum_arr))
            sum_arr_sort_with_idx = sorted(enumerate(sum_arr), key=lambda sum_arr:sum_arr[1])  # sum_arr[1] since in enumerate(sum_arr), value is in position1 while idx is in position0
            # corersponding idx for the sorted sum_arr
            sum_arr_sort_idx = [x[0] for x in sum_arr_sort_with_idx]
            tmp_idx = 1
            while group_job_table[idx] == J_max:
                idx = sum_arr_sort_idx[tmp_idx]
                tmp_idx += 1
                
            avg_arr[idx].extend([v])
            group_idx_table[ordered_seq_len_table[i][5]] = idx
            group_job_table[idx] += 1     

def find_average_with_foundation_method1(new_seq_len_table, KV_buffering_amount_list, die_idx_table, die_job_table, new_seq_len_table_id, J_max=0, J_min=0, MH_alg_distr_mode=0):
    new_seq_len_table_copy = copy.deepcopy(new_seq_len_table)
    
    for i in range(len(new_seq_len_table_copy)):
        new_seq_len_table_copy[i] += (new_seq_len_table_id[i],)
        
    ordered_new_seq_len_table = sorted(new_seq_len_table_copy, key=lambda x:(x[2]), reverse=True)
    
    # for i in ordered_new_seq_len_table:
    #     print(i)
    
    avg_arr = []
    sum_arr = []
    for i in range(len(KV_buffering_amount_list)):
        avg_arr.append([])
        sum_arr.append(0)
    
    if MH_alg_distr_mode == 0:
        # origin distribution
        for i in range(len(ordered_new_seq_len_table)):
            v = ordered_new_seq_len_table[i][2]
            for j in range(len(KV_buffering_amount_list)):
                sum_arr[j] = sum(avg_arr[j]) + KV_buffering_amount_list[j]
            idx = sum_arr.index(min(sum_arr))
            avg_arr[idx].extend([v])
            KV_buffering_amount_list[idx] += v
            die_idx_table[ordered_new_seq_len_table[i][5]] = idx
    else:
        # distribution with boundary
        """ check if there are dies whose job number is below the minimum boundary """
        # table that record die idx with job number below the minimum boundary
        die_with_less_jobs_table_idx = []
        # table that record die data amount with job number below the minimum boundary
        die_with_less_jobs_table = []
        for i in range(len(die_job_table)):
            if die_job_table[i] < J_min:
                die_with_less_jobs_table_idx.append(i)
                die_with_less_jobs_table.append(KV_buffering_amount_list[i])
        
        flag = True 
        if len(die_with_less_jobs_table_idx) >= 1:
            """ if exists dies with job number below the minimum boundary """
            # sum_arr_less = []
            # for i in len(die_with_less_jobs_table_idx):
            #     sum_arr_less.append(0)
                
            ascending_new_seq_len_table = sorted(new_seq_len_table_copy, key=lambda x:(x[2]), reverse=False)
            for i in range(len(ordered_new_seq_len_table)):
                # every time get the present minimum
                v = ascending_new_seq_len_table[0][2]
                # record the amount of every die with job number below the minimum boundary
                for j in range(len(die_with_less_jobs_table_idx)):
                    die_with_less_jobs_table[j] = KV_buffering_amount_list[die_with_less_jobs_table_idx[j]]
                # find die idx with the most amount among dies with job number below the minimum boundary
                idx = die_with_less_jobs_table_idx[die_with_less_jobs_table.index(max(die_with_less_jobs_table))]
                KV_buffering_amount_list[idx] += v
                die_idx_table[ordered_new_seq_len_table[i][5]] = idx
                die_job_table[idx] += 1
                # this value is used so pop it
                ascending_new_seq_len_table.pop(0)
                if check_die_job_table(die_job_table, J_min):
                    # now all die has job number above the minimum boundary, break
                    break
                
            if len(ascending_new_seq_len_table) >= 1:
                # if there's remaining new jobs
                ordered_new_seq_len_table = sorted(ascending_new_seq_len_table, key=lambda x:(x[2]), reverse=True)
            else:
                flag = False
            
        if flag:    
            """ if no die has job number below the minimum boundary now and thereis still new jobs to be assigned"""
            for i in range(len(ordered_new_seq_len_table)):
                v = ordered_new_seq_len_table[i][2]
                for j in range(len(KV_buffering_amount_list)):
                    sum_arr[j] = sum(avg_arr[j]) + KV_buffering_amount_list[j]
                idx = sum_arr.index(min(sum_arr))
                sum_arr_sort_with_idx = sorted(enumerate(sum_arr), key=lambda sum_arr:sum_arr[1])  # sum_arr[1] since in enumerate(sum_arr), value is in position1 while idx is in position0
                # corersponding idx for the sorted sum_arr
                sum_arr_sort_idx = [x[0] for x in sum_arr_sort_with_idx]
                tmp_idx = 1
                while die_job_table[idx] == J_max:
                    idx = sum_arr_sort_idx[tmp_idx]
                    tmp_idx += 1
                    
                avg_arr[idx].extend([v])
                KV_buffering_amount_list[idx] += v
                die_idx_table[ordered_new_seq_len_table[i][5]] = idx
                die_job_table[idx] += 1
                
def check_die_job_table(die_job_table, J_min):
    for i in die_job_table:
        if i < J_min:
            return False
    return True

def MH_latency(die_num, core_num, seq_len_table, group_idx_table, B, H, sram2_height, mac_lane, mac_num, head_embedding_dim, params, 
                    NonLinear_Statistics, p_NonLinear_Statistics, J_max=0, J_min=0, MH_alg_distr_mode=0, MH_alg_level_mode=0, debug_on=False):
    """ 
    seq_len_table: (if this time this job is in prefill stage, requests_id, sequence length, done?)
    p_NonLinear_Statistics[2]: for softmax statistics 
    """
    # FIXME since there's padding overhead for MH, when assign K/V to differnent dies, we may consider s after padding instead of raw s
    
    """ initialization """
    seg_latencys = latency()
    seg_latencys.get_params(params)
    latencys = []
    softmax_loading_amount = []
    
    if debug_f:
        print("---------- MH start ------------")
    
    if MH_alg_level_mode == 0:
        for i in range(die_num):
            latencys.append(latency())
            latencys[i].get_params(params)
            softmax_loading_amount.append(0)
    else:
        for i in range(core_num):
            latencys.append(latency()) 
            latencys[i].get_params(params)
            softmax_loading_amount.append(0)
        
    """ begin MH execution """
    for i in range(B):
        # if debug_f:
        #     print("================ new job =================")
        if seq_len_table[i][3] == False:
            # if this is a live job
            
            # this job is assigned to which core
            group_idx = group_idx_table[i]
            
            # this job's overall generated sequence length
            s = seq_len_table[i][2]
            # if debug_f:
            #     print(f"s: {s}")
            
            # how many rounds(heads) should a group process
            if MH_alg_level_mode == 0:
                # FIXME consider what if H cannot be divided by core_num
                rounds = math.ceil(H / core_num)
            else:
                rounds = math.ceil(H / die_num)
            
            if seq_len_table[i][0]:
                """ prefill stage """
                
                """ MH1 calculation """
                if sram2_height * mac_num < head_embedding_dim * s:
                    # if debug_f:
                    #     print("prefill-MH1, SRAM cannot hold K/V")
                    # if SRAM cannot hold K/V
                    seg_num = math.ceil(s / (sram2_height / math.ceil(head_embedding_dim / mac_num)))
                    weights_col_per_seg = math.ceil(s / seg_num)
                    weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                    (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, s)
                    psum_num = math.ceil(head_embedding_dim / mac_num)

                    for weights_col in weights_col_per_sram_list:
                        for IA_rows in weights_col_per_sram_list:
                            # seg_latencys = matrix_matrix_mul_latency(IA_rows, head_embedding_dim, weights_col, mac_lane, mac_num, psum_num, rounds, 0, debug_on)
                            seg_latencys = matrix_matrix_mul_latency(IA_rows, head_embedding_dim, weights_col, mac_lane, mac_num, psum_num, rounds, 0, (sram2_height * mac_num < (s * s + head_embedding_dim * s)))
                            latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                    
                else:
                    # if debug_f:
                    #     print("prefill-MH1, SRAM can hold K/V")
                    # if SRAM can hold K/V and A
                    psum_num = math.ceil(head_embedding_dim / mac_num)
                    seg_latencys = matrix_matrix_mul_latency(s, head_embedding_dim, s, mac_lane, mac_num, psum_num, rounds, 0, (sram2_height * mac_num < (s * s + head_embedding_dim * s)))
                    # if debug_f:
                    #     seg_latencys.dump_portion("MH-job")
                    # print(f"group_idx: {group_idx}")
                    latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                
                """ MH2 calculation """
                if (sram2_height * mac_num < head_embedding_dim * s) and (sram2_height * mac_num < s * s):
                    # if SRAM cannot hold K/V and A
                    # if debug_f:
                    #     print("prefill-MH2, SRAM cannot hold K/V or A")
                    # FIXME here we assume s is less then 4096
                    seg_num = math.ceil(head_embedding_dim / math.floor(sram2_height / math.ceil(s / mac_num)))    
                    weights_col_per_seg = math.ceil(head_embedding_dim / seg_num)
                    weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                    (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, head_embedding_dim)  
                    
                    seg_num = math.ceil(s / math.floor(sram2_height / math.ceil(s / mac_num)))
                    A_col_per_seg = math.ceil(s / seg_num)
                    A_col_maclane_per_seg = math.ceil(A_col_per_seg / mac_lane)
                    (A_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, A_col_maclane_per_seg, mac_lane, s)
                    
                    psum_num = math.ceil(s / mac_num)
                    for weights_col in weights_col_per_sram_list:
                        for IA_rows in A_col_per_sram_list:
                            seg_latencys = matrix_matrix_mul_latency(IA_rows, s, weights_col, mac_lane, mac_num, psum_num, rounds, 1, (sram2_height * mac_num < (s * s + head_embedding_dim * s)))    
                            latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                            
                elif (sram2_height * mac_num >= head_embedding_dim * s) and (sram2_height * mac_num < s * s):
                    # if SRAM can hold K/V but not A
                    # if debug_f:
                    #     print("prefill-MH2, SRAM can hold K/V but not A")
                    seg_num = math.ceil(s / math.floor(sram2_height / math.ceil(s / mac_num)))
                    A_col_per_seg = math.ceil(s / seg_num)
                    A_col_maclane_per_seg = math.ceil(A_col_per_seg / mac_lane)
                    (A_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, A_col_maclane_per_seg, mac_lane, s)
                    
                    psum_num = math.ceil(s / mac_num)
                    for IA_rows in A_col_per_sram_list:
                        seg_latencys = matrix_matrix_mul_latency(IA_rows, s, head_embedding_dim, mac_lane, mac_num, psum_num, rounds, 1, (sram2_height * mac_num < (s * s + head_embedding_dim * s)))    
                        latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                        
                elif (sram2_height * mac_num < head_embedding_dim * s) and (sram2_height * mac_num >= s * s):
                    # if SRAM can hold A but not K/V
                    # if debug_f:
                    #     print("prefill-MH2, SRAM can hold A but not K/V")
                    seg_num = math.ceil(head_embedding_dim / math.floor(sram2_height / math.ceil(s / mac_num)))    
                    weights_col_per_seg = math.ceil(head_embedding_dim / seg_num)
                    weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                    (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, head_embedding_dim)  
                    psum_num = math.ceil(s / mac_num)

                    for weights_col in weights_col_per_sram_list:
                        # seg_latencys = matrix_matrix_mul_latency(IA_rows, head_embedding_dim, weights_col, mac_lane, mac_num, psum_num, rounds, 0, debug_on)
                        seg_latencys = matrix_matrix_mul_latency(s, s, weights_col, mac_lane, mac_num, psum_num, rounds, 1, (sram2_height * mac_num < (s * s + head_embedding_dim * s)))
                        latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                else:
                    # SRAM can hold K/V and A
                    # if debug_f:
                    #     print("prefill-MH2, SRAM can hold K/V and A")
                    psum_num = math.ceil(s / mac_num)
                    seg_latencys = matrix_matrix_mul_latency(s, s, head_embedding_dim, mac_lane, mac_num, psum_num, rounds, 1, (sram2_height * mac_num < (s * s + head_embedding_dim * s)))
                    # if debug_f:
                    #     seg_latencys.dump_portion("MH-job")
                    latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                
                """ softmax statistics """
                if sram2_height * mac_num < (s * s + head_embedding_dim * s):
                    # MH1 result cannot be hold in SRAM, we should load and store for softmax
                    softmax_loading_amount[group_idx] += s * s * rounds
                    latencys[group_idx].IA_loading += math.ceil(s * s / mac_num) * rounds 
            else:
                """ generation stage """
                if sram2_height * mac_num < head_embedding_dim * s:
                    # SRAM cannot hold K/V
                    # if :debug_f
                        # print("generation-MH1, SRAM cannot hold K/V")
                    
                    # if SRAM2 cannot hold entire K at the same time
                    """ MH1 calculation """
                    seg_num = math.ceil(s / (sram2_height / math.ceil(head_embedding_dim / mac_num)))
                    weights_col_per_seg = math.ceil(s / seg_num)
                    weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                    (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, s)    
                        
                    psum_num = math.ceil(head_embedding_dim / mac_num)
                    ii = 0
                    for weights_col in weights_col_per_sram_list:
                        seg_latencys = vector_matrix_mul_latency(head_embedding_dim, weights_col, mac_lane, mac_num, psum_num, rounds, 0, ii == 0)
                        latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                        ii += 1
                        
                    """ MH2 calcualtion """   
                    # if debug_f:
                    #     print("generation-MH2, SRAM cannot hold K/V")
                    # FIXME here we assume s is less then 4096
                    seg_num = math.ceil(head_embedding_dim / math.floor(sram2_height / math.ceil(s / mac_num)))    
                    weights_col_per_seg = math.ceil(head_embedding_dim / seg_num)
                    weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                    (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, head_embedding_dim)  

                    psum_num = math.ceil(s / mac_num)
                    ii = 0
                    for weights_col in weights_col_per_sram_list:
                        seg_latencys = vector_matrix_mul_latency(s, weights_col, mac_lane, mac_num, psum_num, rounds, 1, ii == 0)    
                        latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                        ii += 1
                else:
                    """ MH1 calculation """
                    # if debug_f:
                    #     print("generation-MH1, SRAM can hold K/V")
                    psum_num = math.ceil(head_embedding_dim / mac_num)
                    seg_latencys = vector_matrix_mul_latency(head_embedding_dim, s, mac_lane, mac_num, psum_num, rounds, 0)
                    latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                    
                    """ MH2 calculation """
                    # if debug_f:
                    #     print("generation-MH2, SRAM can hold K/V")
                    psum_num = math.ceil(s / mac_num)
                    seg_latencys = vector_matrix_mul_latency(s, head_embedding_dim, mac_lane, mac_num, psum_num, rounds, 1)
                    latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
    
    if debug_f:              
        print("---------- MH end ------------")
    
    # FIXME this should be right, no mismatch
    NonLinear_Statistics.latency.IA_loading += math.ceil(max(softmax_loading_amount) / mac_num) * 2
    p_NonLinear_Statistics[2].latency.IA_loading += math.ceil(max(softmax_loading_amount) / mac_num) * 2
    
    # FIXME
    sum = [0] * (die_num if MH_alg_level_mode == 0 else core_num)
    for i in range(len(latencys)):
        sum[i] = latencys[i].overall_latency()
    if debug_on:
        print(f"MH latencys: {sum}")     
        print(f"max latency idx: {sum.index(max(sum))}")
        
    return latencys[sum.index(max(sum))]  

def LayerNorm_latency(IA_rows, IA_cols, die_num, core_num, mac_lane, mac_num, psum_num, sram2_height, params, debug_on=False):
    # suppose at the very beginning, IA is partitioned to store on different dies
    # steps:
    # 1. load data into cores(suppose no NoC), every core has several cols
    # 2. do local(core) average, than NoC and calculate die average(average is 24-bit)
    # 3. NoP and calculate overall average, route the average into dies
    # 4. NoC route the average into cores and every core calculates portion of Var's denominator
    # 5. NoC add the denominator of every core and NoP to get the Var and NoP back to each die
    # 6. NoC the Var to each core and calculate the final result 
    # 7. Write the final result back to the HMC
    # NOTE: 1. How IA is partitioned among dies is decided by FC calculation
    #       2. We use the overall latency of the first die to reperesent the final latency
    
    latencys = latency()
    latencys.get_params(params)
    
    """ rotation mode """
    # number of IA_rows * mac_num * psum_num sub-matrix of IA
    big_col_num = math.ceil(IA_cols / mac_num / psum_num)
    IA_big_col_per_die = math.ceil(big_col_num / die_num)
    IA_core_big_col_per_die = math.ceil(IA_big_col_per_die / core_num)
    # every element in the list can be divided by core number
    IA_big_col_per_die_list = [0] * die_num
    (IA_big_col_per_die_list, _) = calculate_weights_col_assignment(die_num, IA_core_big_col_per_die, core_num, big_col_num)
    
    if debug_f:
        print("---------- LN start ------------")          
        print(f"IA_big_col_per_die_list: {IA_big_col_per_die_list}")  
        print(f"IA_rows: {IA_rows}")
    
    # 1. loading all data to the logic die                # number of big column in each core
    latencys.IA_loading += IA_rows * psum_num * math.ceil(IA_big_col_per_die_list[0] / core_num)
    
    # 2.1 every core calcualtes their local average(addition+division)
    # latencys.add_8 += IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num)
    # latencys.division += IA_rows
    latencys.add_ppu_count(IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num))
    latencys.add_ppu_count(IA_rows)
    
    # 2.2 NoC to die's PPU to calculate die average
    # latencys.NoC_24 += IA_rows 
    # latencys.add_24 += core_num * IA_rows
    # latencys.division += IA_rows
    # for i in range(core_num):
    latencys.add_NoC_cores_to_ppu(IA_rows * 3)
    latencys.add_ppu_count(core_num * IA_rows * 3)
    latencys.add_ppu_count(IA_rows * 3)
    
    # 3.1 NoP to shared PPU
    # latencys.NoP_24 += die_num * IA_rows
    # for i in range(die_num):
    latencys.add_NoP_dies_to_ppu(IA_rows * 3)
    
    # 3.2 calculate the overall average
    # latencys.add_24 += die_num * IA_rows
    # latencys.division += IA_rows
    latencys.add_ppu_count(die_num * IA_rows * 3)
    latencys.add_ppu_count(IA_rows * 3)
    
    # 3.3 NoP to each die
    # latencys.NoP_24 += IA_rows
    # for i in range(die_num):
    latencys.add_NoP_ppu_to_dies_s(IA_rows * 3)
    
    # 4.1 NoC average to each core
    # latencys.NoC_24 += IA_rows
    # for i in range(core_num):
    latencys.add_NoC_ppu_to_cores_s(IA_rows * 3)
    
    # 4.2 each core calculates portion of Var's denominator
    # latencys.substract_square += IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num)
    # latencys.add_24 += IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num)
    # latencys.division += IA_rows
    latencys.add_ppu_count(IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num))
    latencys.add_ppu_count(IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num))
    latencys.add_ppu_count(IA_rows * 3)
    
    # 5.1 NoC to die's PPU to calculate die average
    # latencys.NoC_24 += IA_rows 
    # latencys.add_24 += core_num * IA_rows
    # latencys.division += IA_rows
    # for i in range(core_num):
    latencys.add_NoC_cores_to_ppu(IA_rows * 3)
    latencys.add_ppu_count(core_num * IA_rows * 3)
    latencys.add_ppu_count(IA_rows * 3)
    
    # 5.2 NoP to shared PPU
    # latencys.NoP_24 += die_num * IA_rows
    # for i in range(die_num):
    latencys.add_NoP_dies_to_ppu(IA_rows * 3)
    
    # 5.3 calculate the overall average
    # latencys.add_24 += die_num * IA_rows
    # latencys.division += IA_rows
    latencys.add_ppu_count(die_num * IA_rows * 3)
    latencys.add_ppu_count(IA_rows * 3)
    
    # 5.4 NoP to each die
    # latencys.NoP_24 += IA_rows
    # for i in range(die_num):
    latencys.add_NoP_ppu_to_dies_s(IA_rows * 3)
    
    # 6.1 NoC average to each core
    # latencys.NoC_24 += IA_rows
    # for i in range(core_num):
    latencys.add_NoC_ppu_to_cores_s(IA_rows * 3)
    
    # 6.2 now every core has EX and Var, each can calculate the final results
    # latencys.LN += IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num)
    # FIXME: here LN operation is complicated rather than a single operation, maybe we should add this multiple times
    latencys.add_ppu_count(IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num))
    
    # 7. write back to HMC
    latencys.IA_loading += IA_rows * psum_num * math.ceil(IA_big_col_per_die_list[0] / core_num)
    
    # NOTE: here we can left first round of IA on logic die to save latency
    """ calcualte how many rounds of calculation should a die goes through for IA """
    # number of rounds to complete using IA in row-dim
    rounds_IA = math.ceil(IA_rows / (sram2_height / psum_num))
    IA_rows_per_round = math.ceil(IA_rows / rounds_IA)
    IA_maclane_row_per_round = math.ceil(IA_rows_per_round / mac_lane)
    IA_rows_per_round_list = [0] * rounds_IA
    IA_rows_per_round_max = 0
    (IA_rows_per_round_list, IA_rows_per_round_max) = calculate_weights_col_assignment(rounds_IA, IA_maclane_row_per_round, mac_lane, IA_rows)
    if debug_f:
        print(f"IA_rows_per_round_list: {IA_rows_per_round_list}")
    latencys.IA_loading -= IA_rows_per_round_list[0] * psum_num

    # if debug_on:
    #     latencys.dump()
    if debug_f:
        print("---------- LN end ------------")          
            
    return latencys

def split_latency(die_num, core_num, seq_len_table, group_idx_table, B, weight_cols, IA_rows, mac_lane, mac_num, params, MH_alg_level_mode, J_max=0, J_min=0, MH_alg_distr_mode=0, sim_mode=0, debug_on=False):
    # FIXME: some of IA_loading latency should be neglect since it direclty follows FC calculation and MH directly follows it
    #        but these latency seems negligible 
    # suppose IA is evenly distributed among CHs of memory die 
    # die-level distribution
    # 1. loading IA on logic die(choose the maximum latency among all dies)
    # 2. NoP sending and receiving(choose the maximum latency among all dies)
    # 3. writing IA back to HMC(choose the maximum latency among all dies) 
    
    latencys = latency()
    latencys.get_params(params)
    
    if debug_f:
        print(f"---------- Split start ------------")
    
    """ initialization """
    KV_buffering_amount_list = []
    # record job number within each die/core
    group_job_table = []
    
    # if MH_alg_level_mode == 0:
    for i in range(die_num if MH_alg_level_mode == 0 else core_num):
        KV_buffering_amount_list.append(0)
        group_job_table.append(0)
    # else:
    #     for i in range(core_num):
    #         latencys.append(latency())
    #         KV_buffering_amount_list.append(0)
    #         group_job_table.append(0)
    
    """ record current """
    for i in range(B):
        # record each core's K/V buffering status
        if (seq_len_table[i][0] == False) and (seq_len_table[i][3] == False):
            # if in generation stage(since prompts in prefill stage are new jobs) and this is a live job
            KV_buffering_amount_list[group_idx_table[i]] += seq_len_table[i][2]
            group_job_table[group_idx_table[i]] += 1
    
    """ assign new Q-As's K/V location """
    new_seq_len_table = [x for i, x in enumerate(seq_len_table) if (seq_len_table[i][0] == True)]
    new_seq_len_table_id = [i for i, x in enumerate(seq_len_table) if (seq_len_table[i][0] == True)]
    
    # if debug_on:
    #     # print(f"MH stage, new_seq_len_table: {new_seq_len_table}")
    #     print(f"KV_buffering_amount_list before: {KV_buffering_amount_list}")
    #     print("new seq len table")
    #     for e in new_seq_len_table:
    #         print(e)    

    # if at the very beginning
    if len(new_seq_len_table) == B:
        find_average_method1(seq_len_table, die_num if MH_alg_level_mode == 0 else core_num, group_idx_table, J_max, J_min, 0 if sim_mode == 1 else 1)
        # find_average_method1(seq_len_table, die_num, group_idx_table, J_max, J_min, 1)
    elif len(new_seq_len_table) >= 1:
        find_average_with_foundation_method1(new_seq_len_table, KV_buffering_amount_list, group_idx_table, group_job_table, new_seq_len_table_id, J_max, J_min, MH_alg_distr_mode)
    # if there is no new job, no redistribution is needed
    
    # if debug_on:
    #     print(f"KV_buffering_amount_list after: {KV_buffering_amount_list}")
        
    # print(f"max:{max(KV_buffering_amount_list)}, min:{min(KV_buffering_amount_list)}")
    # print(f"{max(KV_buffering_amount_list)},{min(KV_buffering_amount_list)}")
    
    # IA distribution is decided by FC1's weight partition
    weights_col_per_die = math.ceil(weight_cols / die_num) 
    weights_maclane_col_per_die = math.ceil(weights_col_per_die / mac_lane)
    weights_col_per_die_list = [0] * die_num
    weights_col_per_die_max = 0
    (weights_col_per_die_list, weights_col_per_die_max) = calculate_weights_col_assignment(die_num, weights_maclane_col_per_die, mac_lane, weight_cols)
    
    if debug_f:
        print(f"weight_col_per_die_list: {weights_col_per_die_list}")
    
    if MH_alg_level_mode == 0:
        """ die level distribution """ 

        # how many rows belong to this die
        IA_rows_out = [0] * die_num
        # how many rows should be sent out
        IA_rows_in = [0] * die_num
        # how many 8-bit data belong to this die
        IA_matrix_out = [0] * die_num
        # how many 8-bit data should be sent out
        IA_matrix_in = [0] * die_num
        
        # 1. loading IA
        for i in range(B):
            IA_rows_in[group_idx_table[i]] += 1 if seq_len_table[i][0] == False else seq_len_table[i][2]
        for i in range(die_num):
            IA_rows_out[i] = IA_rows - IA_rows_in[i] 
            IA_matrix_in[i] = math.ceil((weight_cols - weights_col_per_die_list[i]) / mac_num) * IA_rows_in[i]
            IA_matrix_out[i] = IA_rows_out[i] * math.ceil(weights_col_per_die_list[i] / mac_num)
        
        # for i in group_idx_table:
        #     print(i)
        if debug_f:
            print(f"IA_rows_in: {IA_rows_in}")
            print(f"IA_rows_out: {IA_rows_out}")
            print(f"IA_matrix_in: {IA_matrix_in}")
            print(f"IA_matrix_out: {IA_matrix_out}")
        
        latencys.IA_loading += math.ceil(max(IA_matrix_out) / core_num)
        
        average_list = []
        for i in range(die_num):
            average_list.append(math.ceil(IA_matrix_out[i] / (die_num - 1)) * mac_num)
        if debug_f:
            print(f"average_list: {average_list}")
        # 2. NoP, sending & receiving
        # NoC data to PPU
        latencys.add_NoC_cores_to_ppu(math.ceil(max(IA_matrix_out) / core_num) * mac_num)
        # NoP all to all
        latencys.add_NoP_all_to_all(average_list)
        # NoC data back to cores 
        latencys.add_NoC_ppu_to_cores_m(math.ceil(max(IA_matrix_in) / core_num) * mac_num)
        
        # 3. writing IA
        latencys.IA_loading += math.ceil(max(IA_matrix_in) / core_num)
        
        if debug_on:
            print("--------------- Split start -----------------")
            print(f"IA_rows: {IA_rows}")
            print(f"IA_rows_in: {IA_rows_in}")
            print(f"IA_rows_out: {IA_rows_out}")
            print(f"IA_matrix_in: {IA_matrix_in}")
            print(f"IA_matrix_out: {IA_matrix_out}")
            latencys.dump()
            print("--------------- Split end -----------------")
    else:
        """ core level distribution """
        # 1. loading IA
        latencys.IA_loading += math.ceil(IA_rows * max(weights_col_per_die_list) * 2 / 3 / mac_num / core_num) 
        
        average_list = []
        for i in range(die_num):
            average_list.append(IA_rows * weights_col_per_die_list[i] * 2 / 3 / (die_num - 1))
        # 2. NoP
        # NoC data to PPU
        latencys.add_NoC_cores_to_ppu(IA_rows * max(weights_col_per_die_list) * 2 / 3 / core_num)
        # NoP all to all
        latencys.add_NoP_all_to_all(average_list)
        # NoC data back to cores
        latencys.add_NoC_ppu_to_cores_m(IA_rows * max(weights_col_per_die_list) * 2 / 3 / core_num)      
        # latencys.NoP_8 += math.ceil(IA_rows * max(weights_col_per_die_list) * 2 / 3) * 2
         
        # 3. writing IA
        latencys.IA_loading += math.ceil(IA_rows * max(weights_col_per_die_list) * 2 / 3 / mac_num / core_num)
        
        if debug_on:
            print("--------------- Split start -----------------")
            print(f"IA_rows: {IA_rows}")
            print(f"max(weights_col_per_die_list): {max(weights_col_per_die_list)}")
            latencys.dump()
            print("--------------- Split end -----------------")
    if debug_f:
        print(f"---------- Split end ------------")
    return latencys

def softmax_latency(die_num, core_num, seq_len_table, group_idx_table, B, H, head_embedding_dim, mac_lane, params, MH_alg_level_mode=0, debug_on=False):
    # original softmax steps for die-level distribution:
    # 1. upon every mac_lane vec is calculated, we update the local maximum(core) 
    # 2. when the whole vec finishes calculation, local(core) maximum is obtianed
    # 3. NoC the local maximums to die's PPU and find the global maximum, then routes the global maximum back to each core
    # 4. calculate the exp of every element and calculates the local sum
    # 5. NoC the local sums to get the global sum and NoC back to each core
    # 6. calcualte the final results and starts MH2 calculation
    
    # original softmax steps for core-level distribution:
    # 1. upon every mac_lane vec is calculated, we update the local maximum(core) 
    # 2. when the whole vec finishes calculation, local(core) maximum is obtianed
    # 3. NoP the local maximums to shared PPU and find the global maximum, then routes the global maximum back to each die
    # 4. calculate the exp of every element and calculates the local sum
    # 5. NoP the local sums to get the global sum and NoP back to each core
    # 6. calcualte the final results and starts MH2 calculation
    
    latencys = latency()
    latencys.get_params(params)
    IA_rows_per_group_list = [0] * (die_num if MH_alg_level_mode == 0 else core_num)
    rounds = math.ceil(H / (core_num if MH_alg_level_mode == 0 else die_num))
    # number of mac_lane vectors in a head
    # vec_num = math.ceil(head_embedding_dim / mac_lane)
    
    if debug_f:
        print("-------------- softmax start --------------")
    for i in range(B):
        # print(group_idx_table[i])
        IA_rows_per_group_list[group_idx_table[i]] += 1 if (seq_len_table[i][0] == False) else seq_len_table[i][2]
    if debug_f: 
        print(f"IA_rows_per_group_list: {IA_rows_per_group_list}")        
    
    if MH_alg_level_mode == 0:
        """ original softmax, communication within die """
        # 1. find local maximum
        # latencys.find_max += rounds * vec_num
        latencys.add_ppu_count(rounds * head_embedding_dim)
        
        # 3.1 route local maximum to PPU
        latencys.add_NoC_cores_to_ppu(1)
        # 3.2 find global maximum among core_num data
        # latencys.find_max += 1
        latencys.add_ppu_count(core_num)
        # 3.3 route the global maximum back to each core
        latencys.add_NoC_ppu_to_cores_s(1)
        
        # 4.1 calculate the exp of every element
        # latencys.substract_exp += rounds * head_embedding_dim
        # subtract max
        latencys.add_ppu_count(rounds * head_embedding_dim)
        # exp
        latencys.add_ppu_count(rounds * head_embedding_dim)
        # 4.2 calculate the locals sum
        # latencys.add_24 += rounds * head_embedding_dim
        latencys.add_ppu_count(3 * rounds * head_embedding_dim)
        
        # 5.1 route local sum to PPU
        # latencys.NoC_24 += 1 
        latencys.add_NoC_cores_to_ppu(3)
        # 5.2 get global sum
        # latencys.add_24 += core_num
        latencys.add_ppu_count(core_num)
        # 5.3 route global sum to each core
        # latencys.NoC_24 += 1
        latencys.add_NoC_ppu_to_cores_s(3)
        
        # 6. division
        # latencys.division += rounds * head_embedding_dim
        latencys.add_ppu_count(3 * rounds * head_embedding_dim)
            
    elif MH_alg_level_mode == 1:
        """ optimized softmax, communication across die """
        # 1. find local maximum
        # latencys.find_max += rounds * vec_num
        latencys.add_ppu_count(rounds * head_embedding_dim)
        
        # 3.1 route local maximum to PPU
        # latencys.NoP_8 += 1
        latencys.add_NoC_core_to_ppu(1)
        latencys.add_NoP_dies_to_ppu(1)
        # 3.2 find global maximum among die_num data
        # latencys.find_max += 1
        latencys.add_ppu_count(die_num)
        # 3.3 route the global maximum back to each core
        # latencys.NoP_8 += 1
        latencys.add_NoP_ppu_to_dies_s(1)
        latencys.add_NoC_ppu_to_core(1)
        
        # 4.1 calculate the exp of every element
        # latencys.substract_exp += rounds * head_embedding_dim
        # subtract max
        latencys.add_ppu_count(rounds * head_embedding_dim)
        # exp
        latencys.add_ppu_count(rounds * head_embedding_dim)
        # 4.2 calculate the locals sum
        # latencys.add_24 += rounds * head_embedding_dim
        latencys.add_ppu_count(3 * rounds * head_embedding_dim)
        
        # 5.1 route local sum to PPU
        # latencys.NoP_24 += 1 
        latencys.add_NoC_core_to_ppu(3)
        latencys.add_NoP_dies_to_ppu(3)
        # 5.2 get global sum
        # latencys.add_24 += core_num
        latencys.add_ppu_count(die_num)
        # 5.3 route global sum to each core
        # latencys.NoP_24 += 1
        latencys.add_NoP_ppu_to_dies_s(3)
        latencys.add_NoC_ppu_to_core(3)
        
        # 6. division
        # latencys.division += rounds * head_embedding_dim
        latencys.add_ppu_count(3 * rounds * head_embedding_dim)
        
    latencys.multiple(max(IA_rows_per_group_list))

    if debug_on:
        print("------------------ softmax start -------------------")
        # print(f"rounds: {rounds}, vec_num: {vec_num}")
        print(f"IA_rows_per_die_list: {IA_rows_per_group_list}")
        latencys.dump()
    if debug_f:
        print("------------------ softmax end -------------------")
        
    
    return latencys
    
def concat_latency(die_num, core_num, B, seq_len_table, group_idx_table, IA_rows, weight_cols, mac_lane, mac_num, params, MH_alg_level_mode, debug_on=False):
    
    latencys = latency()
    latencys.get_params(params)
    if debug_f:
        print("-------------- Concat start ----------------")
    
    if MH_alg_level_mode == 0:
        # IA distribution is decided by FC2's weight partition
        weights_col_per_die = math.ceil(weight_cols / die_num) 
        weights_maclane_col_per_die = math.ceil(weights_col_per_die / mac_lane)
        weights_col_per_die_list = [0] * die_num
        weights_col_per_die_max = 0
        (weights_col_per_die_list, weights_col_per_die_max) = calculate_weights_col_assignment(die_num, weights_maclane_col_per_die, mac_lane, weight_cols)
        if debug_f:
            print(f"weight_col_per_die_list: {weights_col_per_die_list}")
                
        """ Split """
        # how many rows belong to this die
        IA_rows_out = [0] * die_num
        # how many rows should be sent out
        IA_rows_in = [0] * die_num
        # how many 8-bit data belong to this die
        IA_matrix_out = [0] * die_num
        # how many 8-bit data should be sent out
        IA_matrix_in = [0] * die_num
        
        # 1. loading IA
        for i in range(B):
            IA_rows_out[group_idx_table[i]] += 1 if seq_len_table[i][0] == False else seq_len_table[i][2]
        for i in range(die_num):
            IA_rows_in[i] = IA_rows - IA_rows_out[i] 
            IA_matrix_out[i] = math.ceil((weight_cols - weights_col_per_die_list[i]) / mac_num) * IA_rows_out[i]
            IA_matrix_in[i] = IA_rows_in[i] * math.ceil(weights_col_per_die_list[i] / mac_num)
        
        latencys.IA_loading += math.ceil(max(IA_matrix_out) / core_num)
        
        average_list = []
        for i in range(die_num):
            average_list.append(math.ceil(IA_matrix_out[i] / (die_num - 1)) * mac_num)
        # 2. NoP, sending & receiving
        # NoC data to PPU
        latencys.add_NoC_cores_to_ppu(math.ceil(max(IA_matrix_out) / core_num) * mac_num)
        # NoP all to all
        latencys.add_NoP_all_to_all(average_list)
        # NoC data back to cores 
        latencys.add_NoC_ppu_to_cores_m(math.ceil(max(IA_matrix_in) / core_num) * mac_num)
        # latencys.NoP_8 += (max(IA_matrix_out) + max(IA_matrix_in)) * mac_num
        
        # 3. writing IA
        latencys.IA_loading += math.ceil(max(IA_matrix_in) / core_num)
        
        if debug_f:
            print(f"IA_rows_in: {IA_rows_in}")
            print(f"IA_rows_out: {IA_rows_out}")
            print(f"IA_matrix_in: {IA_matrix_in}")
            print(f"IA_matrix_out: {IA_matrix_out}")
            print(f"average_list: {average_list}")
            
        if debug_on:
            print("------------------ Concat start -------------------")
            print(f"weight_col_per_die_list: {weights_col_per_die_list}")
            print(f"IA_rows: {IA_rows}")
            print(f"weight_cols: {weight_cols}")
            print(f"IA_row_in: {IA_rows_in}")
            print(f"IA_row_out: {IA_rows_out}")
            print(f"IA_matrix_in: {IA_matrix_in}")
            print(f"IA_matrix_out: {IA_matrix_out}")
            latencys.dump()
            print("------------------ Concat end -------------------")
    else:
        # core level distribution, no concat routing is needed
        pass
    
    if debug_f:
        print("-------------- Concat end ----------------")    
    return latencys
          
def dump_configs(requests_pool, requests_total_count, die_num, mac_lane, mac_num, sram2_height, N, B, H, D, embedding_dim, head_embedding_dim, MH_alg_level_mode, J_max=0, J_min=0, MH_alg_distr_mode=0):
    print("================ HW configs =================")
    print(f"+ die number: {die_num}")
    print(f"+ core number in a die: {N}")
    print(f"+ mac lane number: {mac_lane}")
    print(f"+ mac num number: {mac_num}")
    print(f"+ sram2_height: {sram2_height}")
    print("================ SW configs =================")
    print(f"+ embedding dimension: {embedding_dim}")
    print(f"+ head number: {H}")
    print(f"+ head_embedding dimension: {head_embedding_dim}")
    print(f"+ decoder block number: {D}")
    print("================ MH algorithm ===============")
    print(f"+ MH algorithm level mode: {MH_alg_level_mode}")
    print(f"+ MH distribution mode: {MH_alg_distr_mode}")
    if MH_alg_distr_mode == 1:
        print(f"+ J_max: {J_max}")
        print(f"+ J_min: {J_min}")
    print("================= Data set ==================")
    print(f"+ batch size: {B}")
    print(f"+ request_total_count:{requests_total_count}")
    # print("+ requests_pool")
    # for e in requests_pool:
    #     print(e)

def simulation(args, requests_pool, Statistics, FC_Statistics, MH_Statistics, NonLinear_Statistics, p_NonLinear_Statistics):
    """ 
    requests_pool: (used, prompt_len, answer len)
    """
    
    """ HW """
    die_num = args.die_num
    mac_lane = args.mac_lane
    mac_num = args.mac_num
    sram2_height = args.SRAM2_height
    N = args.core_num_in_die
    
    """ MAPPING """
    W = args.W
    # 0 for IA rotation across dies, 1 for IA duplication, every die has complete IA
    # softmax_mode = args.softmax_mode
 
    """ SW """
    B = args.batch_size
    # B = 8
    H = args.head_num
    D = args.decoder_num
    embedding_dim = args.embedding_dim
    head_embedding_dim = embedding_dim // H
        
    # how many requests are there in the requests pool - for simulation purpose
    requests_total_count = len(requests_pool)
    
    # record each Q-A's present sequence length
    # (if this time this job is in prefill stage, requests_id, seqeunce length, done?, is recomputation?)
    # done is only for the last requets marking whether they completes
    seq_len_table = [(True, 0, 0, False, False)] * B
    
    # record each Q-A's assigned die idx when MH_alg_level_mode == 0
    die_idx_table = [0] * B
    # record each Q-A's assigned core idx when MH_alg_level_mode == 1
    core_idx_table = [0] * B
    
    """ algorithm """
    P = args.P
    MH_alg_distr_mode = args.MH_alg_distr_mode
    MH_alg_level_mode = args.MH_alg_level_mode
    if MH_alg_level_mode == 0:
        # ideal job number per die
        J_ideal = math.ceil(B / die_num)
        # maximum job number per die 
        J_max = math.ceil((1 + P) * J_ideal)
        # minimum job nubmer per die
        J_min = math.floor((1 - P) * J_ideal)
    else:
        J_ideal = math.ceil(B / N)
        J_max = math.ceil((1 + P) * J_ideal)
        J_min = math.floor((1 - P) * J_ideal)
    
    print("================================ Initial Start ===============================")
    dump_configs(requests_pool, requests_total_count, die_num, mac_lane, mac_num, sram2_height, N, B, H, D, 
                    embedding_dim, head_embedding_dim, MH_alg_level_mode, J_max, J_min, MH_alg_distr_mode)

    """ seq_len_table initiation """
    for i in range(B):
        seq_len_table[i] = [True, i, requests_pool[i][1], False, False]
        requests_pool[i][0] = True
    # record which job will be feed into the system next time
    requests_id = B

    # if args.sim_mode == 0:
    #     print("seq_len_table:")
    #     for e in seq_len_table:
    #         print(e)
            
    # rows of input activation height
    IA_rows = 0
    
    """ LATENCY """
    seg_latency = latency()
    nonlinear_latencys = latency()
    total_latency = 0

    # psum_vec_cal_latency = args.psum_vec_cal_latency
    # psum_cube_cal_latency = args.psum_cube_cal_latency
    # psum_cube_cal_N_latency = args.psum_cube_cal_N_latency
    # psum_cube_loading_latency = args.psum_cube_loading_latency
    # add_8_latency = args.add_8_latency
    # add_24_latency = args.add_8_latency 
    # substract_square_latency = args.substract_square_latency
    # division_latency = args.division_latency
    # LN_latency = args.LN_latency
    # find_max_latency = args.find_max_latency
    # substract_exp_latency = args.substract_exp_latency
    # IA_loading_latency = args.weight_loading_latency
    # psum_cube_wb_latency = args.psum_cube_wb_latency
    # csum_vec_wb_latency = args.csum_vec_wb_latency
    # csum_vec_wb_sram_latency = args.csum_vec_wb_sram_latency
    # NoC_8_latency = args.avg_NoC_8_latency
    # NoC_24_latency = args.avg_NoC_8_latency  
    # NoP_8_latency = args.avg_NoP_8_latency
    # NoP_24_latency = args.avg_NoP_8_latency 
    params = [args.cal_latency, args.ppu_latency, args.ppu_boundary, args.weight_loading_latency, args.NoC_core_bandwidth, args.NoP_die_bandwidth, args.ppu_bandwidth, \
                args.NoC_latency_per_hop, args.NoP_latency_per_hop, args.NoC_h, args.NoC_w, args.NoP_h, args.NoP_w, args.is_ring_mode]
    assert(N == args.NoC_h * args.NoC_w)
    assert(die_num == args.NoP_h * args.NoP_w)
    
    Statistics.latency.get_params(params)
    FC_Statistics.latency.get_params(params)
    MH_Statistics.latency.get_params(params)
    NonLinear_Statistics.latency.get_params(params)
    for i in p_NonLinear_Statistics: 
        i.latency.get_params(params)
    Statistics.latency.dump_params()
    
    """ memory usage """
    Statistics.peak_static_mem = D * 12 * math.pow(embedding_dim, 2)     
    Statistics.peak_IA = 3 * 4 * math.pow(embedding_dim, 2)            
     
    """ debug """
    debug_on = False
    # debug_on_simple = False
    once = False
    
    # print(f"requests_id:{requests_id}")
    print("================================ Initial End ===============================")
    
    rounds = 0
    mh_count = 0
    
    
    if args.sim_mode == 0:
        IA_rows_sum = 0
        flag2 = False
        once = True
        while requests_done(seq_len_table) == False:
            rounds += 1
            if debug_f:
                print("======================================= new round =======================================")
                # for e in seq_len_table:
                #     print(e)
            # if flag:
                # print(f"remaining requests: {requests_total_count - requests_id}")
                
            """ profiling """
            IA_rows = 0
            for e in seq_len_table:
                if e[0]:
                    # if in prefill stage
                    IA_rows += e[2]
                    # mh_count += math.pow(e[2],2)
                    mh_count += e[2]
                else:
                    # in generation stage
                    IA_rows += 1
                    mh_count += e[2]
            IA_rows_sum += IA_rows
                    
            if debug_on:
                print(f"IA rows: {IA_rows}")
                
            """ LN1 """
            seg_latency = LayerNorm_latency(IA_rows, embedding_dim, die_num, N, mac_lane, mac_num, W, sram2_height, params, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, NonLinear_Statistics, D)
            latency_acc1(seg_latency, p_NonLinear_Statistics[0], D)
            if debug_f:
                seg_latency.dump_portion("SEG")
                Statistics.latency.dump_portion("total")
                NonLinear_Statistics.latency.dump_portion("nonlinear")
                p_NonLinear_Statistics[0].latency.dump_portion("0")
                        
            """ FC1 """
            # if debug_on:
            #     print("------------- FC1 -------------")
            # seg_latency = stage_latency(die_num, N, 3 * embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            # NOTE suppose at the very beginning of
            (seg_latency, nonlinear_latency) = stage_latency(die_num, N, 3 * embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, params, 1, True, False)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, FC_Statistics, D)
            latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
            latency_substract(FC_Statistics, nonlinear_latency)
            if debug_f:
                seg_latency.dump_portion("SEG")
                Statistics.latency.dump_portion("total")
                NonLinear_Statistics.latency.dump_portion("nonlinear")
                FC_Statistics.latency.dump_portion("FC1")
            # if debug_on:
            #     seg_latency.dump()
            #     Statistics.dump()
            
            """ Split """
            seg_latency = split_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, B, 
                                            3 * embedding_dim, IA_rows, mac_lane, mac_num, params, MH_alg_level_mode, J_max, J_min, MH_alg_distr_mode, args.sim_mode)   
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, NonLinear_Statistics, D)
            latency_acc1(seg_latency, p_NonLinear_Statistics[1], D)
            if debug_f:
                seg_latency.dump_portion("SEG")
                Statistics.latency.dump_portion("total")
                NonLinear_Statistics.latency.dump_portion("nonlinear")
                p_NonLinear_Statistics[1].latency.dump_portion("1")
            
                    
            """ MH """
            # Scale is done immediately after MH1
            # if debug_on:
            #     print("------------- MH -------------")
            seg_latency = MH_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, B, H, 
                                        sram2_height, mac_lane, mac_num, head_embedding_dim, params, NonLinear_Statistics, p_NonLinear_Statistics, J_max, J_min, MH_alg_distr_mode, MH_alg_level_mode)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, MH_Statistics, D)
            if debug_f:
                seg_latency.dump_portion("SEG")
                Statistics.latency.dump_portion("total")
                MH_Statistics.latency.dump_portion("MH")
                p_NonLinear_Statistics[2].latency.dump_portion("2")
            # if debug_on:
            #     seg_latency.dump()
            #     Statistics.dump()
            
            """ Softmax """
            seg_latency = softmax_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, B, H, 
                                            head_embedding_dim, mac_lane, params, MH_alg_level_mode, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, NonLinear_Statistics, D)
            latency_acc1(seg_latency, p_NonLinear_Statistics[2], D)
            if debug_f:
                seg_latency.dump_portion("SEG")
                Statistics.latency.dump_portion("total")
                NonLinear_Statistics.latency.dump_portion("nonlinear")
                p_NonLinear_Statistics[2].latency.dump_portion("2")
            
            """ Concat """
            seg_latency = concat_latency(die_num, N, B, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, IA_rows, embedding_dim, mac_lane, mac_num, params, MH_alg_level_mode)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, NonLinear_Statistics, D)
            latency_acc1(seg_latency, p_NonLinear_Statistics[3], D)
            if debug_f:
                seg_latency.dump_portion("SEG")
                p_NonLinear_Statistics[3].latency.dump_portion("3")
            
            """ FC2 """
            # if debug_on:
            #     print("------------- FC2 -------------")
            # seg_latency = stage_latency(die_num, N, embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            (seg_latency, nonlinear_latency) = stage_latency(die_num, N, embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, params, 2, False, True)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, FC_Statistics, D)
            latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
            latency_substract(FC_Statistics, nonlinear_latency)
            if debug_f:
                seg_latency.dump_portion("SEG")
                Statistics.latency.dump_portion("total")
                NonLinear_Statistics.latency.dump_portion("nonlinear")
                FC_Statistics.latency.dump_portion("FC2")
            # if debug_on:
            #     seg_latency.dump()
            #     Statistics.dump()
            
            """ LN2 """
            seg_latency = LayerNorm_latency(IA_rows, embedding_dim, die_num, N, mac_lane, mac_num, W, sram2_height, params)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, NonLinear_Statistics, D)
            latency_acc1(seg_latency, p_NonLinear_Statistics[0], D)
            if debug_f:
                seg_latency.dump_portion("SEG")
                Statistics.latency.dump_portion("total")
                NonLinear_Statistics.latency.dump_portion("nonlinear")
                p_NonLinear_Statistics[0].latency.dump_portion("0")
            
            """ FC3 """
            # if debug_on:
            #     print("------------- FC3 -------------")
            # seg_latency = stage_latency(die_num, N, 4 * embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            (seg_latency, nonlinear_latency) = stage_latency(die_num, N, 4 * embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, params, 3, True, False)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, FC_Statistics, D)
            latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
            latency_substract(FC_Statistics, nonlinear_latency)
            if debug_f:
                seg_latency.dump_portion("SEG")
                Statistics.latency.dump_portion("total")
                NonLinear_Statistics.latency.dump_portion("nonlinear")
                FC_Statistics.latency.dump_portion("FC3")
            # if debug_on:
            #     seg_latency.dump()
            #     Statistics.dump()
            
            """ FC4 """
            # if debug_on:
            #     print("------------- FC4 -------------")
            # seg_latency = stage_latency(die_num, N, embedding_dim, 4 * embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            (seg_latency, nonlinear_latency) = stage_latency(die_num, N, embedding_dim, 4 * embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, params, 4, False, True)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, FC_Statistics, D)
            latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
            latency_substract(FC_Statistics, nonlinear_latency)
            if debug_f:
                seg_latency.dump_portion("SEG")
                Statistics.latency.dump_portion("total")
                NonLinear_Statistics.latency.dump_portion("nonlinear")
                FC_Statistics.latency.dump_portion("FC4")
            # if debug_on:
            #     seg_latency.dump()
            #     Statistics.dump()
            
            """ at the end of each round, update dynamic weight """
            # length in sequence length table decides the dynamic memory, which has nothing to do with prefill stage or not
            tmp_dynamic_weight = 0
            for i in range(B):
                if (seq_len_table[i][3] == False) and (seq_len_table[i][4] == False):
                    # if this is a live job and is not recomputation
                    tmp_dynamic_weight += seq_len_table[i][2]
            tmp_dynamic_weight = 2 * embedding_dim * D * tmp_dynamic_weight
            Statistics.peak_dynamic_mem = max(tmp_dynamic_weight, Statistics.peak_dynamic_mem)
            
            debug_on = False
            flag1 = False
            group_jobs = [0] * (die_num if MH_alg_level_mode == 0 else N)
            """ at the end of the round, check if certain Q-A stops """
            for i in range(B):
                # if flag1 == False:
                #     for ii in range(B):
                #         if seq_len_table[ii][3] == False:
                #             group_jobs[die_idx_table[ii] if MH_alg_level_mode == 0 else core_idx_table[ii]] += 1
                #     print(f"group_jobs: {group_jobs}")
                #     print('finished jobs:')
                #     flag1 = True
                    
                if seq_len_table[i][0] == False:
                    # in generation stage
                    # answer length ++
                    if seq_len_table[i][3] == False:
                        seq_len_table[i][2] += 1
                    if seq_len_table[i][2] == (requests_pool[seq_len_table[i][1]][1] + requests_pool[seq_len_table[i][1]][2]):
                        # if this job's generation completes, feed a new job into batch
                        # print(f"length: {seq_len_table[i][2]}, group{die_idx_table[i] if MH_alg_level_mode == 0 else core_idx_table[i]}")
                        if requests_id == requests_total_count:
                            # requests pool is run out
                            seq_len_table[i][3] = True
                            if flag2 == False:
                                print("------------------------------------")
                                flag2 = True
                        else:
                            assert (requests_pool[requests_id][0] == False)   # this must be an unused request 
                            seq_len_table[i] = [True, requests_id, requests_pool[requests_id][1], False, False]
                            requests_pool[requests_id][0] = True
                            requests_id += 1         
                            # debug_on = True  
                            # debug_on_simple = True
                            flag = True
                else:
                    # in prefill stage, next time switch to generation stage
                    seq_len_table[i][0] = False
                    
            group_jobs = [0] * (die_num if MH_alg_level_mode == 0 else N)
        
            # if rounds >= 3:
            #     break
            # if once:
            #     break
            # for ii in range(B):
            #     if (seq_len_table[ii][3] == False) and (seq_len_table[ii][0] == False):
            #         group_jobs[die_idx_table[ii] if MH_alg_level_mode == 0 else core_idx_table[ii]] += 1
            # print(f"group_jobs after stopping: {group_jobs}")
            
        # print(f"IA_rows_sum: {IA_rows_sum}")
        # print(f"rounds: {rounds}")
        # print(f"mh_count: {mh_count}")
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
        IA_rows_list = []
        for b in B_list:
            IA_rows_sum = 0
            # find the max answer/prompt length to pad with
            requests_pool_seg = requests_pool[B_idx * B: (B_idx + 1) * B if b == B else requests_total_count]
            max_prompt_len = max(requests_pool_seg, key=itemgetter(1))[1] 
            max_answer_len = max(requests_pool_seg, key=itemgetter(2))[2] 
            # print("----------------------------- new round ---------------------------")
            # print(f"B_idx: {B_idx}")
            # print(f"[max_prompt_len, max_answer_len]: [{max_prompt_len}, {max_answer_len}]")
            
            for i in range(b):
                seq_len_table[i] = [True, i, max_prompt_len, False, False]
            seq_len_table = seq_len_table[:b]    
            
            # print("seq_len_table:")
            # for e in seq_len_table:
            #     print(e)
            
            while seq_len_table[0][2] < (max_prompt_len + max_answer_len):
                rounds += 1
                
                if seq_len_table[0][0] or (seq_len_table[0][2] == (max_prompt_len + 1)):
                    debug_on = True
                    
                # if debug_on:
                #     print("--------------- new round ----------------")
                #     for e in seq_len_table:
                #         print(e)
                
                """ profiling """
                IA_rows = 0
                for e in seq_len_table:
                    if e[0]:
                        # if in prefill stage
                        IA_rows += e[2]
                    else:
                        # in generation stage
                        IA_rows += 1
                IA_rows_sum += IA_rows
                # if debug_on:
                #     print(f"IA rows: {IA_rows}")
                         
                """ LN1 """
                seg_latency = LayerNorm_latency(IA_rows, embedding_dim, die_num, N, mac_lane, mac_num, W, sram2_height, params)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, NonLinear_Statistics, D)
                latency_acc1(seg_latency, p_NonLinear_Statistics[0], D)
                
                """ FC1 """
                # if r > 0:
                #     # FIXME Detail this 
                #     latency += IA_latency
                # latency += (H - 1) * IA_latency # FIXME
                # if debug_on:
                #     print("------------- FC1 -------------")
                # seg_latency = stage_latency(die_num, N, 3 * embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
                (seg_latency, nonlinear_latency) = stage_latency(die_num, N, 3 * embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, params, 1, True, False)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, FC_Statistics, D)
                latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
                latency_substract(FC_Statistics, nonlinear_latency)
                # if debug_on:
                #     seg_latency.dump()
                #     Statistics.dump()
                
                """ Split """
                seg_latency = split_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, b, 
                                                3 * embedding_dim, IA_rows, mac_lane, mac_num, params, MH_alg_level_mode, 0, 0, 0, args.sim_mode)   
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, NonLinear_Statistics, D)      
                latency_acc1(seg_latency, p_NonLinear_Statistics[1], D)
                
                """ MH """
                # latency += H * IA_latency # FIXME
                # if debug_on:
                #     print("------------- MH -------------")
                seg_latency =  MH_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, b, H, 
                                        sram2_height, mac_lane, mac_num, head_embedding_dim, params, NonLinear_Statistics, p_NonLinear_Statistics, 0, 0, 0, MH_alg_level_mode)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, MH_Statistics, D)
                # if debug_on:
                #     seg_latency.dump()
                #     Statistics.dump()
                
                """ Softmax """
                seg_latency = softmax_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, b, H, 
                                                head_embedding_dim, mac_lane, params, MH_alg_level_mode)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, NonLinear_Statistics, D)
                latency_acc1(seg_latency, p_NonLinear_Statistics[2], D)
                
                """ Concat """
                seg_latency = concat_latency(die_num, N, b, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, IA_rows, embedding_dim, mac_lane, mac_num, params, MH_alg_level_mode)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, NonLinear_Statistics, D)
                latency_acc1(seg_latency, p_NonLinear_Statistics[3], D)
                
                """ FC2 """
                # latency += H * IA_latency # FIXME
                # if debug_on:
                #     print("------------- FC2 -------------")
                # seg_latency = stage_latency(die_num, N, embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
                (seg_latency, nonlinear_latency) = stage_latency(die_num, N, embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, params, 2, False, True)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, FC_Statistics, D)
                latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
                latency_substract(FC_Statistics, nonlinear_latency)
                # if debug_on:
                #     seg_latency.dump()
                #     Statistics.dump()
                
                """ LN2 """
                seg_latency = LayerNorm_latency(IA_rows, embedding_dim, die_num, N, mac_lane, mac_num, W, sram2_height, params)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, NonLinear_Statistics, D)
                latency_acc1(seg_latency, p_NonLinear_Statistics[0], D)
                
                """ FC3 """
                # latency += H * IA_latency # FIXME
                # if debug_on:
                #     print("------------- FC3 -------------")
                # seg_latency = stage_latency(die_num, N, 4 * embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
                (seg_latency, nonlinear_latency) = stage_latency(die_num, N, 4 * embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, params, 2, True, False)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, FC_Statistics, D)
                latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
                latency_substract(FC_Statistics, nonlinear_latency)
                # if debug_on:
                #     seg_latency.dump()
                #     Statistics.dump()
                
                """ FC4 """
                # latency += H * IA_latency # FIXME
                # if debug_on:
                #     print("------------- FC4 -------------")
                # seg_latency = stage_latency(die_num, N, embedding_dim, 4 * embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
                (seg_latency, nonlinear_latency) = stage_latency(die_num, N, embedding_dim, 4 * embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, params, 2, False, True)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, FC_Statistics, D)
                latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
                latency_substract(FC_Statistics, nonlinear_latency)
                # if debug_on:
                #     seg_latency.dump()
                #     Statistics.dump()
                
                """ at the end of each round, update dynamic weight """
                # length in sequence length table decides the dynamic memory, which has nothing to do with prefill stage or not
                tmp_dynamic_weight = 0
                for i in range(b):
                    tmp_dynamic_weight += seq_len_table[i][2]
                tmp_dynamic_weight = 2 * embedding_dim * D * tmp_dynamic_weight
                Statistics.peak_dynamic_mem = max(tmp_dynamic_weight, Statistics.peak_dynamic_mem)
            
                for e in seq_len_table:
                    if e[0]:
                        # mh_count += math.pow(e[2],2)
                        mh_count += e[2]
                        e[0] = False
                    else:
                        mh_count += e[2]
                    e[2] += 1
                        
                        
                debug_on = False

                
            IA_rows_list.append(IA_rows_sum)
            B_idx += 1
            
        print(f"IA_rows_list: {IA_rows_list}")
        print(f"rounds: {rounds}")
        print(f"mh_count: {mh_count}")
               
    total_latency = Statistics.latency.overall_latency()
    
    fc_latency = FC_Statistics.latency.overall_latency()
    
    mh_latency = MH_Statistics.latency.overall_latency()
    
    nonlinear_latency = NonLinear_Statistics.latency.overall_latency()
    
    nonlinear_latencys = []
    
    for i in p_NonLinear_Statistics:
        nonlinear_latencys.append(i.latency.overall_latency())
    print("done")
    
    return (total_latency, fc_latency, mh_latency, nonlinear_latency, nonlinear_latencys)
        
def main():
    """ Main function """
    
    args = argparser().parse_args()
    (requests_pool, s_avg) = init_requests_pool()
    # print(f"average requests and answer length: {s_avg}")
    Statistics = Stats()
    FC_Statistics = Stats()
    MH_Statistics = Stats()
    NonLinear_Statistics = Stats()
    
    LN_Statistics = Stats()  #0
    Split_Statistics = Stats()  #1
    Softmax_Statistics = Stats()  #2
    Concat_Statistics = Stats()  #3
    Residual_Statistics = Stats()  #4
    Rotation_Statistics = Stats()  #5
    (total_latency, fc_latency, mh_latency, nonlinear_latency, nonlinear_latencys) = simulation(args, requests_pool, Statistics, FC_Statistics, MH_Statistics, NonLinear_Statistics,  
                                                                                                [LN_Statistics, Split_Statistics, Softmax_Statistics, Concat_Statistics, Residual_Statistics, Rotation_Statistics])
    dump_res(total_latency, fc_latency, mh_latency, nonlinear_latency, nonlinear_latencys, 
                FC_Statistics, MH_Statistics, NonLinear_Statistics, [LN_Statistics, Split_Statistics, Softmax_Statistics, Concat_Statistics, Residual_Statistics, Rotation_Statistics], Statistics)
        
    return 0
    
if __name__ == '__main__':
    main()