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
8. handle to overlapped of loading and writing IA between different stages
9. add stats to  record the peak memory usage
"""

""" 
NOTE
1. no FC-IA mode, since we choose weight partition, the result maxtrix(next layer's IA) is naturally partitioned in dies
2. no softmax mode, since we can do local maximum, data amount required for routing is the same as linear-approximate version
3. core-level and die-level distribution, the latency difference lies in whether concat is needed
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
    
    # MAPPING
    ap.add_argument('--W', type = int, default = 1,
                    help = 'parameter when calculating FC layer, for every core, determine the width of dot production, width = mac_num * W')
    # ap.add_argument('--softmax-mode', type = int, default = 0,
    #                 help = '0 for original softmax, 1 for optimized softmax')
    
    # MODEL
    ap.add_argument('--batch-size', type = int, default = 1024,
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
    ap.add_argument('--MH-alg-level-mode', type = int, default = 1,
                    help = '0 for implementation the load-balance algorithm in die level, 1 for core level')

    # PARAM
    ap.add_argument('--cal-latency', type = int, default = 0.5,
                    help = 'latency of calculating a mac_num dot production(8bit)')
    ap.add_argument('--psum-vec-cal-latency', type = int, default = 0.5,
                    help = 'latency of adding 2 mac_lane * 3Byte data')
    ap.add_argument('--psum-cube-cal-latency', type = int, default = 0.5,
                    help = 'latency of adding 2 mac_lane * mac_lane * 3Byte data')
    ap.add_argument('--psum-cube-cal-N-latency', type = int, default = 0.5,
                    help = 'latency of adding N(core number in a die) mac_lane * mac_lane * 3Byte data')
    ap.add_argument('--psum-cube-loading-latency', type = int, default = 0.5,
                    help = 'latency of reading a mac_lane * mac_lane * 3Byte data from HMC')
    ap.add_argument('--add-8-latency', type = int, default = 0.5,
                    help = 'latency of adding 2 8-bit data together')
    ap.add_argument('--add-24-latency', type = int, default = 0.5,
                    help = 'latency of adding 2 24-bit data together')
    ap.add_argument('--substract-square-latency', type = int, default = 0.5,
                    help = 'latency of adding 2 (oprand1 - oprand2)^2, one oprand is 8-bit and the other is 24-bit')
    ap.add_argument('--division-latency', type = int, default = 0.5,
                    help = 'latency of divide two data, one is 24-bit, one is 8-bit')
    ap.add_argument('--LN-latency', type = int, default = 0.5,
                    help = 'latency of (x-EX)/(sqrt(Var+epsilon))*gamma + beta')
    ap.add_argument('--find_max_latency', type = int, default = 0.5,
                    help = 'latency of finding maximum among mac_lane data')
    ap.add_argument('--substract_exp_latency', type = int, default = 0.5,
                    help = 'latency of e^(x-x_max) of a data')
    ap.add_argument('--weight-loading-latency', type = int, default = 0.5,
                    help = 'latency of transfering mac_num Byte of data from HMC to SRAM of core')
    ap.add_argument('--psum-cube-wb-latency', type = int, default = 0.5,
                    help = 'latency of writing a mac_lane * mac_lane * 3Byte data to HMC')
    ap.add_argument('--csum-vec-wb-latency', type = int, default = 0.5,
                    help = 'latency of writing a complete mac_lane Byte data to HMC')
    ap.add_argument('--csum-vec-wb-sram-latency', type = int, default = 0.5,
                    help = 'latency of writing a complete mac_lane Byte data to SRAM of core')
    ap.add_argument('--avg-NoC-8-latency', type = int, default = 0.5,
                    help = 'average latency of transferring a mac_num Byte data between cores or from one die to the shared PPU in the die')
    ap.add_argument('--avg-NoP-8-latency', type = int, default = 0.5,
                    help = 'average latency of transferring a mac_num Byte data between dies or from one die to the shared PPU')
    
    # OTHERS
    ap.add_argument('--debug-flag', type = bool, default = True)       
    ap.add_argument('--sim-mode', type = int, default = 1,
                    help = '0 for my design, 1 for FlexGen prompt/answer padding implementation')
    
    return ap

def dump_res(total_latency, fc_latency, mh_latency, nonlinear_latency, nonlinear_latencys, FC_Statistics, MH_Statistics, NonLinear_Statistics, p_NonLinear_Statistics, Statistics):
    print("================================ Results ===============================")
    Statistics.dump("Total")
    FC_Statistics.dump("FC")
    MH_Statistics.dump("MH")
    NonLinear_Statistics.dump("NonLinear")
    
    p_NonLinear_Statistics[0].dump("Layer Norm")
    p_NonLinear_Statistics[1].dump("Split")
    p_NonLinear_Statistics[2].dump("Softmax")
    p_NonLinear_Statistics[3].dump("Concat")
    p_NonLinear_Statistics[4].dump("Residual")
    p_NonLinear_Statistics[5].dump("IA Rotation")
    
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
    Statistics.latency.psum_vec_cal += stage_latency.psum_vec_cal * D
    Statistics.latency.psum_cube_cal += stage_latency.psum_cube_cal * D
    Statistics.latency.psum_cube_cal_N += stage_latency.psum_cube_cal_N * D
    Statistics.latency.psum_cube_loading += stage_latency.psum_cube_loading * D
    Statistics.latency.add_8 += stage_latency.add_8 * D
    Statistics.latency.add_24 += stage_latency.add_24 * D
    Statistics.latency.substract_square += stage_latency.substract_square * D
    Statistics.latency.division += stage_latency.division * D
    Statistics.latency.LN += stage_latency.LN * D
    Statistics.latency.find_max += stage_latency.find_max * D
    Statistics.latency.substract_exp += stage_latency.substract_exp * D
    Statistics.latency.weight_loading += stage_latency.weight_loading * D
    Statistics.latency.IA_loading += stage_latency.IA_loading * D
    Statistics.latency.psum_cube_wb += stage_latency.psum_cube_wb * D
    Statistics.latency.vec_wb += stage_latency.vec_wb * D
    Statistics.latency.vec_wb_sram += stage_latency.vec_wb_sram * D
    Statistics.latency.NoC_8 += stage_latency.NoC_8 * D
    Statistics.latency.NoC_24 += stage_latency.NoC_24 * D
    Statistics.latency.NoP_8 += stage_latency.NoP_8 * D
    Statistics.latency.NoP_24 += stage_latency.NoP_24 * D
    
def latency_substract(FC_Statistics, nonlinear_latency):
    FC_Statistics.latency.cal -= nonlinear_latency.cal
    FC_Statistics.latency.psum_vec_cal -= nonlinear_latency.psum_vec_cal
    FC_Statistics.latency.psum_cube_cal -= nonlinear_latency.psum_cube_cal
    FC_Statistics.latency.psum_cube_cal_N -= nonlinear_latency.psum_cube_cal_N
    FC_Statistics.latency.psum_cube_loading -= nonlinear_latency.psum_cube_loading
    FC_Statistics.latency.add_8 -= nonlinear_latency.add_8
    FC_Statistics.latency.add_24 -= nonlinear_latency.add_24
    FC_Statistics.latency.substract_square -= nonlinear_latency.substract_square
    FC_Statistics.latency.division -= nonlinear_latency.division
    FC_Statistics.latency.LN -= nonlinear_latency.LN
    FC_Statistics.latency.find_max -= nonlinear_latency.find_max
    FC_Statistics.latency.substract_exp -= nonlinear_latency.substract_exp
    FC_Statistics.latency.weight_loading -= nonlinear_latency.weight_loading
    FC_Statistics.latency.IA_loading -= nonlinear_latency.IA_loading
    FC_Statistics.latency.psum_cube_wb -= nonlinear_latency.psum_cube_wb
    FC_Statistics.latency.vec_wb -= nonlinear_latency.vec_wb
    FC_Statistics.latency.vec_wb_sram -= nonlinear_latency.vec_wb_sram
    FC_Statistics.latency.NoC_8 -= nonlinear_latency.NoC_8
    FC_Statistics.latency.NoC_24 -= nonlinear_latency.NoC_24
    FC_Statistics.latency.NoP_8 -= nonlinear_latency.NoP_8
    FC_Statistics.latency.NoP_24 -= nonlinear_latency.NoP_24
    
def latency_acc2(stage_latency, latencys):
    new_latencys = latency()
    
    new_latencys.cal = latencys.cal + stage_latency.cal
    new_latencys.psum_vec_cal = latencys.psum_vec_cal + stage_latency.psum_vec_cal
    new_latencys.psum_cube_cal = latencys.psum_cube_cal + stage_latency.psum_cube_cal
    new_latencys.psum_cube_cal_N = latencys.psum_cube_cal_N + stage_latency.psum_cube_cal_N
    new_latencys.psum_cube_loading = latencys.psum_cube_loading + stage_latency.psum_cube_loading
    new_latencys.add_8 = latencys.add_8 + stage_latency.add_8
    new_latencys.add_24 = latencys.add_24 + stage_latency.add_24
    new_latencys.substract_square = latencys.substract_square + stage_latency.substract_square
    new_latencys.division = latencys.division + stage_latency.division
    new_latencys.LN = latencys.LN + stage_latency.LN
    new_latencys.find_max = latencys.find_max + stage_latency.find_max
    new_latencys.substract_square = latencys.substract_exp + stage_latency.substract_exp
    new_latencys.weight_loading = latencys.weight_loading + stage_latency.weight_loading
    new_latencys.IA_loading = latencys.IA_loading + stage_latency.IA_loading
    new_latencys.psum_cube_wb = latencys.psum_cube_wb + stage_latency.psum_cube_wb
    new_latencys.vec_wb = latencys.vec_wb + stage_latency.vec_wb
    new_latencys.vec_wb_sram = latencys.vec_wb_sram + stage_latency.vec_wb_sram
    new_latencys.NoC_8 = latencys.NoC_8 + stage_latency.NoC_8
    new_latencys.NoC_24 = latencys.NoC_24 + stage_latency.NoC_24
    new_latencys.NoP_8 = latencys.NoP_8 + stage_latency.NoP_8
    new_latencys.NoP_24 = latencys.NoP_24 + stage_latency.NoP_24
    
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
    
def stage_latency(die_num, core_num, weight_cols, weight_rows, mac_lane, mac_num, psum_num, sram2_height, IA_rows, p_NonLinear_Statistics, is_residual_after=False, debug_on=False):
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
    nonlinear_latencys = latency()
    
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
    for e in IA_big_col_per_die_list:
        IA_rounds_per_die_list.append(math.ceil(e / core_num))
    
    
    """ calculate the maximum weight columns assigned to die """
    weights_col_per_die = math.ceil(weight_cols / die_num) 
    weights_maclane_col_per_die = math.ceil(weights_col_per_die / mac_lane)
    weights_col_per_die_list = [0] * die_num
    weights_col_per_die_max = 0
    (weights_col_per_die_list, weights_col_per_die_max) = calculate_weights_col_assignment(die_num, weights_maclane_col_per_die, mac_lane, weight_cols)
    
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
    
    # if debug_on:
    #     print(f"FC stage, IA_rows_per_round_list: {IA_rows_per_round_list}")    
    
    # FIXME Dummy implementation. We should consider corner cases under special circumstances
    sub_rounds = math.ceil(math.ceil(weight_rows / (mac_num * psum_num)) / core_num)

    # if debug_on: 
    #     print(f"FC stage, rounds_weight: {rounds_weight}")
    #     print(f"FC stage, sub-rounds: {sub_rounds}")
    #     print(f"FC stage, rounds_IA: {rounds_IA}")
    
    # if FC_IA_mode == 1:
    #     """ duplication """
    #     for i in range(rounds_weight):
    #         for j in range(sub_rounds):
    #             # load weights
    #             if weights_col_per_round_list[i] >= IA_rows_per_round_list[0]:
    #                 latencys.weight_loading += weights_col_per_round_list[i] * psum_num
    #             else:
    #                 latencys.IA_loading += IA_rows_per_round_list[0] * psum_num
                
    #             for k in range(rounds_IA):
    #                 # loading IA, but do not need to load weights
    #                 if k > 0:
    #                     latencys.IA_loading += IA_rows_per_round_list[k] * psum_num
                    
    #                 # calculate
    #                 # number of mac-lane col in a calculation block
    #                 maclane_col_num = weights_col_per_round_list[i] / mac_lane
    #                 # number of mac-lane row in a calculation block
    #                 maclane_row_num = IA_rows_per_round_list[k] / mac_lane
                    
    #                 if debug_on:
    #                     print(f"FC stage, in a calculation block, [maclane_col_num, maclane_row_num]: [{maclane_col_num}, {maclane_row_num}]")
                    
    #                 latencys.cal += maclane_col_num * maclane_row_num * mac_lane * psum_num
    #                 latencys.psum_cube_cal += (psum_num - 1) * maclane_col_num * maclane_row_num
    #                 if j > 0:
    #                     # if this is not the first sub-round, we should get partial sum from previous sub-rounds and add to partial sum of this sub-round
    #                     latencys.psum_cube_loading += maclane_col_num * maclane_row_num
    #                 if (j == sub_rounds - 1) and is_residual_after:
    #                     # if this is the last sub-round and there's a residual connection afertwards
    #                     latencys.IA_loading += maclane_col_num * maclane_row_num
                        
    #                 # add N/(N+1)/N+2 cubes together
    #                 latencys.psum_cube_cal_N += maclane_col_num * maclane_row_num
    #                 # write mac-lane * mac-lane block back into HMC
    #                 latencys.psum_cube_wb += maclane_col_num * maclane_row_num 
                

    """ rotation """
    # record which partition of IA in the first die is now processing
    partition_idx = 0
    tmp = 0
    for i in IA_rounds_per_die_list:
        for j in range(rounds_weight):
            for k in range(i):
                if weights_col_per_round_list[j] >= IA_rows_per_round_list[0]:
                    latencys.weight_loading += weights_col_per_round_list[j] * psum_num
                else:
                    if (partition_idx > 0) and (k == 0):
                        # since rotation will transfer data from other die onto the logic die, first subround of IA 
                        latencys.weight_loading += weights_col_per_round_list[j] * psum_num
                    else:
                        latencys.IA_loading += IA_rows_per_round_list[0] * psum_num
                    
                for l in range(rounds_IA):
                    # laoding IA, but do not need to load weights
                    if (l > 0) and ((k > 0) or (partition_idx == 0)):
                        latencys.IA_loading += IA_rows_per_round_list[l] * psum_num
                        
                    # calculate
                    # number of mac-lane col in a calculation block
                    maclane_col_num = weights_col_per_round_list[j] / mac_lane
                    # number of mac-lane row in a calculation block
                    maclane_row_num = IA_rows_per_round_list[l] / mac_lane
                    
                    latencys.cal += maclane_col_num * maclane_row_num * mac_lane * psum_num
                    latencys.psum_cube_cal += (psum_num - 1) * maclane_col_num * maclane_row_num 
                    
                    if k > 0 or partition_idx > 0:
                        # if this is not the first sub-round, we should get partial sum from previous sub-rounds and add to partial sum of this sub-round
                        latencys.psum_cube_loading += maclane_col_num * maclane_row_num
                    
                    if (partition_idx == (len(IA_rounds_per_die_list) - 1)) and (k == (i - 1)) and is_residual_after:
                        # if this is the last sub-round and there's a residual connection afterwards
                        latencys.IA_loading += maclane_col_num * maclane_row_num
                        p_NonLinear_Statistics[4].latency.IA_loading += maclane_col_num * maclane_row_num
                        nonlinear_latencys.IA_loading += maclane_col_num * maclane_row_num
                        tmp += maclane_col_num * maclane_row_num
                        
                    # add N/(N+1)/N+2 cubes together
                    latencys.psum_cube_cal_N += maclane_col_num * maclane_row_num
                    # write mac-lane * mac-lane block back into HMC
                    latencys.psum_cube_wb += maclane_col_num * maclane_row_num 
        
        # a partition of IA finishes calculating, we need to rotate partitions among dies
        # step1: read all the partition data from HMC to logic die
        # step2: transfer to other die's logic die
        # step3: write all the partition data from logic die to HMC
        # NOTE: first sub-round of a partition moves to calculation immediately without writing into HMC first     
        # FIXME: here we use the first die's latency to represent the total latency, however, there exists case that the first die is not the die with the longest processing latency 
        if partition_idx + 1 < len(IA_rounds_per_die_list):          
            latencys.IA_loading += IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows
            p_NonLinear_Statistics[5].latency.IA_loading += IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows
            nonlinear_latencys.IA_loading += IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows
            # sending & receriving
            latencys.NoP_8 += IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows + IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows
            p_NonLinear_Statistics[5].latency.NoP_8 += IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows + IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows
            nonlinear_latencys.NoP_8 += IA_big_col_per_die_list[partition_idx] * psum_num * IA_rows + IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows
            # the first sub-round's data don't need to store 
            latencys.IA_loading += max(IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows - core_num * psum_num * IA_rows, 0)
            p_NonLinear_Statistics[5].latency.IA_loading += max(IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows - core_num * psum_num * IA_rows, 0)
            nonlinear_latencys.IA_loading += max(IA_big_col_per_die_list[partition_idx + 1] * psum_num * IA_rows - core_num * psum_num * IA_rows, 0)
            
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
            
    if debug_on: 
        print("-------------------- Residual Connections start --------------------")             
        print(f"Residual IA loading: {tmp}")
        print("-------------------- Residual Connections end --------------------")             
            
    # latencys.dump()                    
    return (latencys, nonlinear_latencys)

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

def find_average_method1(seq_len_table, group_num, group_idx_table, J_max=0, J_min=0, MH_alg_distr_mode=0):
    # print(f"group_num: {group_num}")
    seq_len_table_copy = copy.deepcopy(seq_len_table)
    for i in range(len(seq_len_table_copy)):
        seq_len_table_copy[i] += (i,)
        
    ordered_seq_len_table = sorted(seq_len_table_copy, key=lambda x:(x[2]), reverse=True)
    
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
            group_idx_table[ordered_seq_len_table[i][4]] = idx
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
            group_idx_table[ordered_seq_len_table[i][4]] = idx
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
            die_idx_table[ordered_new_seq_len_table[i][4]] = idx
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
                die_idx_table[ordered_new_seq_len_table[i][4]] = idx
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
                die_idx_table[ordered_new_seq_len_table[i][4]] = idx
                die_job_table[idx] += 1
                
def check_die_job_table(die_job_table, J_min):
    for i in die_job_table:
        if i < J_min:
            return False
    return True

def MH_latency(die_num, core_num, seq_len_table, group_idx_table, B, H, sram2_height, mac_lane, mac_num, head_embedding_dim, J_max=0, J_min=0, MH_alg_distr_mode=0, MH_alg_level_mode=0, debug_on=False):
    """ 
    seq_len_table: (if this time this job is in prefill stage, requests_id, sequence length, done?)
    """
    # FIXME since there's padding overhead for MH, when assign K/V to differnent dies, we may consider s after padding instead of raw s
    
    """ initialization """
    seg_latencys = latency()
    latencys = []
    
    if MH_alg_level_mode == 0:
        for i in range(die_num):
            latencys.append(latency())
    else:
        for i in range(core_num):
            latencys.append(latency()) 
        
    """ begin MH execution """
    for i in range(B):
        if seq_len_table[i][3] == False:
            # if this is a live job
            
            # this job is assigned to which core
            group_idx = group_idx_table[i]
            
            # this job's overall generated sequence length
            s = seq_len_table[i][2]
            
            # how many rounds(heads) should a group process
            if MH_alg_level_mode == 0:
                # FIXME consider what if H cannot be divided by core_num
                rounds = math.ceil(H / core_num)
            else:
                rounds = math.ceil(H / die_num)
            
            if seq_len_table[i][0]:
                """ prefill stage """
                # FIXME  Should consider the case that SRAM can hold K/V but not twice the capacity of s*s matrix
                
                """ MH1 calculation """
                if sram2_height * mac_num < head_embedding_dim * s:
                    # if SRAM cannot hold K/V
                    seg_num = math.ceil(s / (sram2_height / math.ceil(head_embedding_dim / mac_num)))
                    weights_col_per_seg = math.ceil(s / seg_num)
                    weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                    (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, s)
                    psum_num = math.ceil(head_embedding_dim / mac_num)

                    for weights_col in weights_col_per_sram_list:
                        # FIXME
                        for IA_rows in weights_col_per_sram_list:
                            # seg_latencys = matrix_matrix_mul_latency(IA_rows, head_embedding_dim, weights_col, mac_lane, mac_num, psum_num, rounds, 0, debug_on)
                            seg_latencys = matrix_matrix_mul_latency(IA_rows, head_embedding_dim, weights_col, mac_lane, mac_num, psum_num, rounds, 0)
                            latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                else:
                    # if SRAM can hold K/V
                    # FIXME Here we assume head_embedding_dim can be fully divided by mac_num
                        
                    psum_num = math.ceil(head_embedding_dim / mac_num)
                    seg_latencys = matrix_matrix_mul_latency(s, head_embedding_dim, s, mac_lane, mac_num, psum_num, rounds, 0, (sram2_height * mac_num >= 2 * head_embedding_dim * s))
                    print(f"group_idx: {group_idx}")
                    latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                
                """ MH2 calculation """
                if sram2_height * mac_num < head_embedding_dim * s:
                    # FIXME consider the case cannot hold A
                    # if SRAM cannot hold K/V
                    # FIXME here we assume s is less then 4096
                    seg_num = math.ceil(head_embedding_dim / math.floor(sram2_height / math.ceil(s / mac_num)))    
                    weights_col_per_seg = math.ceil(head_embedding_dim / seg_num)
                    weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                    (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, head_embedding_dim)  
                    
                    psum_num = math.ceil(s / mac_num)
                    for weights_col in weights_col_per_sram_list:
                        for IA_rows in weights_col_per_sram_list:
                        # FIXME
                            seg_latencys = matrix_matrix_mul_latency(IA_rows, s, weights_col, mac_lane, mac_num, psum_num, rounds, 1)    
                            latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                else:
                    psum_num = math.ceil(s / mac_num)
                    seg_latencys = matrix_matrix_mul_latency(s, s, head_embedding_dim, mac_lane, mac_num, psum_num, rounds, 1, True)
                    latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
            else:
                """ generation stage """
                if sram2_height * mac_num < head_embedding_dim * s:
                    # SRAM cannot hold K/V
                    
                    # if SRAM2 cannot hold entire K at the same time
                    """ MH1 calculation """
                    seg_num = math.ceil(s / (sram2_height / math.ceil(head_embedding_dim / mac_num)))
                    weights_col_per_seg = math.ceil(s / seg_num)
                    weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                    (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, s)    
                        
                    psum_num = math.ceil(head_embedding_dim / mac_num)
                    for weights_col in weights_col_per_sram_list:
                        # FIXME
                        seg_latencys = vector_matrix_mul_latency(head_embedding_dim, weights_col, mac_lane, mac_num, psum_num, rounds, 0)
                        latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                        
                    """ MH2 calcualtion """   
                    # FIXME here we assume s is less then 4096
                    seg_num = math.ceil(head_embedding_dim / math.floor(sram2_height / math.ceil(s / mac_num)))    
                    weights_col_per_seg = math.ceil(head_embedding_dim / seg_num)
                    weights_maclane_per_seg = math.ceil(weights_col_per_seg / mac_lane)
                    (weights_col_per_sram_list, _) = calculate_weights_col_assignment(seg_num, weights_maclane_per_seg, mac_lane, head_embedding_dim)  

                    psum_num = math.ceil(s / mac_num)
                    for weights_col in weights_col_per_sram_list:
                        # FIXME
                        seg_latencys = vector_matrix_mul_latency(s, weights_col, mac_lane, mac_num, psum_num, rounds, 1)    
                        latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                else:
                    """ MH1 calculation """
                    # FIXME Here we assume head_embedding_dim can be fully divided by mac_num
                    psum_num = math.ceil(head_embedding_dim / mac_num)
                    seg_latencys = vector_matrix_mul_latency(head_embedding_dim, s, mac_lane, mac_num, psum_num, rounds, 0)
                    latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
                    
                    """ MH2 calculation """
                    psum_num = math.ceil(s / mac_num)
                    seg_latencys = vector_matrix_mul_latency(s, head_embedding_dim, mac_lane, mac_num, psum_num, rounds, 1)
                    latencys[group_idx] = latency_acc2(seg_latencys, latencys[group_idx])
               
    # FIXME
    sum = [0] * (die_num if MH_alg_level_mode == 0 else core_num)
    for i in range(len(latencys)):
        sum[i] = latencys[i].sum() 
    if debug_on:
        print(f"MH latencys: {sum}")     
        print(f"max latency idx: {sum.index(max(sum))}")
        
    return latencys[sum.index(max(sum))]  

def LayerNorm_latency(IA_rows, IA_cols, die_num, core_num, mac_num, psum_num, debug_on=False):
    # suppose at the very beginning, IA is partitioned to store on different dies
    # steps:
    # 1. load data into cores(suppose no NoC), every core has several cols
    # 2. do local(core) average, than NoC and calculate die average(average is 24-bit)
    # 3. NoP and calculate overall average, route the average into dies
    # 4. NoC route the average into cores and every core calculates portion of Var's denominator
    # 5. NoC add the denominator of every core and NoP to get the Var and NoP back to each die
    # 6. NoC the Var to each core and calculate the final result 
    # 7. Write the final result back to the HMC
    # FIXME: 1. part of the final result can be left on the logic die for the upcoming FC calculation, we should substract this part of writing back and laoding latency 
    # TODO: 1. duplication mode should be supported
    # NOTE: 1. How IA is partitioned among dies is decided by FC calculation
    #       2. We use the overall latency of the first die to reperesent the final latency
    
    latencys = latency()
    
    """ rotation mode """
    # number of IA_rows * mac_num * psum_num sub-matrix of IA
    big_col_num = math.ceil(IA_cols / mac_num / psum_num)
    IA_big_col_per_die = math.ceil(big_col_num / die_num)
    IA_core_big_col_per_die = math.ceil(IA_big_col_per_die / core_num)
    # every element in the list can be divided by core number
    IA_big_col_per_die_list = [0] * die_num
    (IA_big_col_per_die_list, _) = calculate_weights_col_assignment(die_num, IA_core_big_col_per_die, core_num, big_col_num)
    
    if debug_on:
        print("---------- LN start ------------")          
        print(f"IA_big_col_per_die_list: {IA_big_col_per_die_list}")  
        print(f"IA_rows: {IA_rows}")
    
    # 1. loading all data to the logic die
    latencys.IA_loading += IA_rows * psum_num * IA_big_col_per_die_list[0]
    
    # 2.1 every core calcualtes their local average
    latencys.add_8 += IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num)
    latencys.division += IA_rows
    
    # 2.2 NoC to die's PPU to calculate die average
    latencys.NoC_24 += IA_rows 
    latencys.add_24 += core_num * IA_rows
    latencys.division += IA_rows
    
    # 3.1 NoP to shared PPU
    latencys.NoP_24 += die_num * IA_rows
    
    # 3.2 calculate the overall average
    latencys.add_24 += die_num * IA_rows
    latencys.division += IA_rows
    
    # 3.3 NoP to each die
    latencys.NoP_24 += IA_rows
    
    # 4.1 NoC average to each core
    latencys.NoC_24 += IA_rows
    
    # 4.2 each core calculates portion of Var's denominator
    latencys.substract_square += IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num)
    latencys.add_24 += IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num)
    latencys.division += IA_rows
    
    # 5.1 NoC to die's PPU to calculate die average
    latencys.NoC_24 += IA_rows 
    latencys.add_24 += core_num * IA_rows
    latencys.division += IA_rows
    
    # 5.2 NoP to shared PPU
    latencys.NoP_24 += die_num * IA_rows
    
    # 5.3 calculate the overall average
    latencys.add_24 += die_num * IA_rows
    latencys.division += IA_rows
    
    # 5.4 NoP to each die
    latencys.NoP_24 += IA_rows
    
    # 6.1 NoC average to each core
    latencys.NoC_24 += IA_rows
    
    # 6.2 now every core has EX and Var, each can calculate the final results
    latencys.LN += IA_rows * psum_num * mac_num * math.ceil(IA_big_col_per_die_list[0] / core_num)
    
    # 7. write back to HMC
    latencys.IA_loading += IA_rows * psum_num * IA_big_col_per_die_list[0]

    
    if debug_on:
        latencys.dump()
        print("---------- LN end ------------")          
            
    return latencys

def split_latency(die_num, core_num, seq_len_table, group_idx_table, B, weight_cols, IA_rows, mac_lane, mac_num, MH_alg_level_mode, J_max=0, J_min=0, MH_alg_distr_mode=0, sim_mode=0, debug_on=False):
    # FIXME: some of IA_loading latency should be neglect since it direclty follows FC calculation and MH directly follows it
    #        but these latency seems negligible 
    # suppose IA is evenly distributed among CHs of memory die 
    # die-level distribution
    # 1. loading IA on logic die(choose the maximum latency among all dies)
    # 2. NoP sending and receiving(choose the maximum latency among all dies)
    # 3. writing IA back to HMC(choose the maximum latency among all dies) 
    
    latencys = latency()
    
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
    print(f"{max(KV_buffering_amount_list)},{min(KV_buffering_amount_list)}")
    
    # IA distribution is decided by FC1's weight partition
    weights_col_per_die = math.ceil(weight_cols / die_num) 
    weights_maclane_col_per_die = math.ceil(weights_col_per_die / mac_lane)
    weights_col_per_die_list = [0] * die_num
    weights_col_per_die_max = 0
    (weights_col_per_die_list, weights_col_per_die_max) = calculate_weights_col_assignment(die_num, weights_maclane_col_per_die, mac_lane, weight_cols)
    
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
        
        latencys.IA_loading += max(IA_matrix_out)
        
        # 2. NoP
        latencys.NoP_8 += (max(IA_matrix_out) + max(IA_matrix_in)) * mac_num
        
        # 3. writing IA
        latencys.IA_loading += max(IA_matrix_in)
        
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
        latencys.IA_loading += math.ceil(IA_rows * max(weights_col_per_die_list) * 2 / 3 / mac_num) 
        
        # 2. NoP
        latencys.NoP_8 += math.ceil(IA_rows * max(weights_col_per_die_list) * 2 / 3) * 2
         
        # 3. writing IA
        latencys.IA_loading += math.ceil(IA_rows * max(weights_col_per_die_list) * 2 / 3 / mac_num)
        
        if debug_on:
            print("--------------- Split start -----------------")
            print(f"IA_rows: {IA_rows}")
            print(f"max(weights_col_per_die_list): {max(weights_col_per_die_list)}")
            latencys.dump()
            print("--------------- Split end -----------------")
        
    return latencys

def softmax_latency(die_num, core_num, seq_len_table, group_idx_table, B, H, head_embedding_dim, mac_lane, MH_alg_level_mode=0, debug_on=False):
    # original softmax steps for die-level distribution:
    # 1. upon every mac_lane vec is calculated, we update the local maximum(core) 
    # 2. when the whole vec finishes calculation, local(core) maximum is obtianed
    # 3. NoC the local maximums to die's PPU and find the global maximum, then routes the global maximum back to each core
    # 4. calculate the exp of every element and calculates the local sum
    # 5. NoC the local sums to get the global sum and NoC back to each core
    # 6. calcualte the final results and starts MH2 calculation
    # FIXME no writing back to HMC and loading from HMC is needed here
    
    # original softmax steps for core-level distribution:
    # 1. upon every mac_lane vec is calculated, we update the local maximum(core) 
    # 2. when the whole vec finishes calculation, local(core) maximum is obtianed
    # 3. NoP the local maximums to shared PPU and find the global maximum, then routes the global maximum back to each die
    # 4. calculate the exp of every element and calculates the local sum
    # 5. NoP the local sums to get the global sum and NoP back to each core
    # 6. calcualte the final results and starts MH2 calculation
    # FIXME no writing back to HMC and loading from HMC is needed here
    
    latencys = latency()
    IA_rows_per_die_list = [0] * (die_num if MH_alg_level_mode == 0 else core_num)
    rounds = math.ceil(H / (core_num if MH_alg_level_mode == 0 else die_num))
    # number of mac_lane vectors in a head
    vec_num = math.ceil(head_embedding_dim / mac_lane)
    
    for i in range(B):
        # print(group_idx_table[i])
        IA_rows_per_die_list[group_idx_table[i]] += 1 if (seq_len_table[i][0] == False) else seq_len_table[i][2]
            
    if MH_alg_level_mode == 0:
        """ original softmax, communication within die """
        # 1. find local maximum
        latencys.find_max += rounds * vec_num
        
        # 3.1 route local maximum to PPU
        latencys.NoC_8 += 1
        # 3.2 find global maximum among core_num data
        # FIXME here we find max out of core_num number of data, but find_max means finding max out of mac_lane number of data
        latencys.find_max += 1
        # 3.3 route the global maximum back to each core
        latencys.NoC_8 += 1
        
        # 4.1 calculate the exp of every element
        latencys.substract_exp += rounds * head_embedding_dim
        # 4.2 calculate the locals sum
        latencys.add_24 += rounds * head_embedding_dim
        
        # 5.1 route local sum to PPU
        latencys.NoC_24 += 1 
        # 5.2 get global sum
        latencys.add_24 += core_num
        # 5.3 route global sum to each core
        latencys.NoC_24 += 1
        
        # 6. division
        # FIXME: division stands for 24-bit/8-bit, hut here is 24-bit/24-bit
        latencys.division += rounds * head_embedding_dim
            
    elif MH_alg_level_mode == 1:
        """ optimized softmax, communication across die """
        # 1. find local maximum
        latencys.find_max += rounds * vec_num
        
        # 3.1 route local maximum to PPU
        latencys.NoP_8 += 1
        # 3.2 find global maximum among die_num data
        # FIXME here we find max out of core_num number of data, but find_max means finding max out of mac_lane number of data
        latencys.find_max += 1
        # 3.3 route the global maximum back to each core
        latencys.NoP_8 += 1
        
        # 4.1 calculate the exp of every element
        latencys.substract_exp += rounds * head_embedding_dim
        # 4.2 calculate the locals sum
        latencys.add_24 += rounds * head_embedding_dim
        
        # 5.1 route local sum to PPU
        latencys.NoP_24 += 1 
        # 5.2 get global sum
        latencys.add_24 += core_num
        # 5.3 route global sum to each core
        latencys.NoP_24 += 1
        
        # 6. division
        # FIXME: division stands for 24-bit/8-bit, hut here is 24-bit/24-bit
        latencys.division += rounds * head_embedding_dim
        
    latencys.multiple(max(IA_rows_per_die_list))
           
    if debug_on:
        print("------------------ softmax start -------------------")
        print(f"rounds: {rounds}, vec_num: {vec_num}")
        print(f"IA_rows_per_die_list: {IA_rows_per_die_list}")
        latencys.dump()
        print("------------------ softmax end -------------------")
        
    
    return latencys
    
def concat_latency(die_num, B, seq_len_table, group_idx_table, IA_rows, weight_cols, mac_lane, mac_num, MH_alg_level_mode, debug_on=False):
    
    latencys = latency()
    
    if MH_alg_level_mode == 0:
        # IA distribution is decided by FC2's weight partition
        weights_col_per_die = math.ceil(weight_cols / die_num) 
        weights_maclane_col_per_die = math.ceil(weights_col_per_die / mac_lane)
        weights_col_per_die_list = [0] * die_num
        weights_col_per_die_max = 0
        (weights_col_per_die_list, weights_col_per_die_max) = calculate_weights_col_assignment(die_num, weights_maclane_col_per_die, mac_lane, weight_cols)
            
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
        
        latencys.IA_loading += max(IA_matrix_out)
        
        # 2. NoP
        latencys.NoP_8 += (max(IA_matrix_out) + max(IA_matrix_in)) * mac_num
        
        # 3. writing IA
        latencys.IA_loading += max(IA_matrix_in)
        
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
    # (if this time this job is in prefill stage, requests_id, seqeunce length, done?)
    # done is only for the last requets marking whether they completes
    seq_len_table = [(True, 0, 0, False)] * B
    
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
    dump_configs(requests_pool, requests_total_count, die_num, mac_lane, mac_num, sram2_height, N, B, H, D, embedding_dim, head_embedding_dim, MH_alg_level_mode, J_max, J_min, MH_alg_distr_mode)

    """ seq_len_table initiation """
    for i in range(B):
        seq_len_table[i] = [True, i, requests_pool[i][1], False]
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
    
    cal_latency = args.cal_latency
    psum_vec_cal_latency = args.psum_vec_cal_latency
    psum_cube_cal_latency = args.psum_cube_cal_latency
    psum_cube_cal_N_latency = args.psum_cube_cal_N_latency
    psum_cube_loading_latency = args.psum_cube_loading_latency
    add_8_latency = args.add_8_latency
    add_24_latency = args.add_8_latency # FIXME
    substract_square_latency = args.substract_square_latency
    division_latency = args.division_latency
    LN_latency = args.LN_latency
    find_max_latency = args.find_max_latency
    substract_exp_latency = args.substract_exp_latency
    weight_loading_latency = args.weight_loading_latency
    IA_loading_latency = args.weight_loading_latency
    psum_cube_wb_latency = args.psum_cube_wb_latency
    csum_vec_wb_latency = args.csum_vec_wb_latency
    csum_vec_wb_sram_latency = args.csum_vec_wb_sram_latency
    NoC_8_latency = args.avg_NoC_8_latency
    NoC_24_latency = args.avg_NoC_8_latency  # FIXME
    NoP_8_latency = args.avg_NoP_8_latency
    NoP_24_latency = args.avg_NoP_8_latency  # FIXME
    
    Statistics.latency.get_params(cal_latency, psum_vec_cal_latency, psum_cube_cal_latency, psum_cube_cal_N_latency, psum_cube_loading_latency, add_8_latency, add_24_latency, substract_square_latency, division_latency, LN_latency,
                   find_max_latency, substract_exp_latency, weight_loading_latency, IA_loading_latency, psum_cube_wb_latency, csum_vec_wb_latency, csum_vec_wb_sram_latency, NoC_8_latency, NoC_24_latency, NoP_8_latency, NoP_24_latency)
    FC_Statistics.latency.get_params(cal_latency, psum_vec_cal_latency, psum_cube_cal_latency, psum_cube_cal_N_latency, psum_cube_loading_latency, add_8_latency, add_24_latency, substract_square_latency, division_latency, LN_latency,
                   find_max_latency, substract_exp_latency, weight_loading_latency, IA_loading_latency, psum_cube_wb_latency, csum_vec_wb_latency, csum_vec_wb_sram_latency, NoC_8_latency, NoC_24_latency, NoP_8_latency, NoP_24_latency)
    MH_Statistics.latency.get_params(cal_latency, psum_vec_cal_latency, psum_cube_cal_latency, psum_cube_cal_N_latency, psum_cube_loading_latency, add_8_latency, add_24_latency, substract_square_latency, division_latency, LN_latency,
                   find_max_latency, substract_exp_latency, weight_loading_latency, IA_loading_latency, psum_cube_wb_latency, csum_vec_wb_latency, csum_vec_wb_sram_latency, NoC_8_latency, NoC_24_latency, NoP_8_latency, NoP_24_latency)
    NonLinear_Statistics.latency.get_params(cal_latency, psum_vec_cal_latency, psum_cube_cal_latency, psum_cube_cal_N_latency, psum_cube_loading_latency, add_8_latency, add_24_latency, substract_square_latency, division_latency, LN_latency,
                   find_max_latency, substract_exp_latency, weight_loading_latency, IA_loading_latency, psum_cube_wb_latency, csum_vec_wb_latency, csum_vec_wb_sram_latency, NoC_8_latency, NoC_24_latency, NoP_8_latency, NoP_24_latency)
    for i in p_NonLinear_Statistics:
        i.latency.get_params(cal_latency, psum_vec_cal_latency, psum_cube_cal_latency, psum_cube_cal_N_latency, psum_cube_loading_latency, add_8_latency, add_24_latency, substract_square_latency, division_latency, LN_latency,
                   find_max_latency, substract_exp_latency, weight_loading_latency, IA_loading_latency, psum_cube_wb_latency, csum_vec_wb_latency, csum_vec_wb_sram_latency, NoC_8_latency, NoC_24_latency, NoP_8_latency, NoP_24_latency)
    
    """ memory usage """
    Statistics.peak_static_mem = D * 12 * math.pow(embedding_dim, 2)     
    Statistics.peak_IA = 3 * 4 * math.pow(embedding_dim, 2)            
     
    """ debug """
    debug_on = False
    # debug_on_simple = False
    flag = True
    
    # print(f"requests_id:{requests_id}")
    print("================================ Initial End ===============================")
    
    rounds = 0
    mh_count = 0
    
    if args.sim_mode == 0:
        IA_rows_sum = 0
        flag2 = False
        while requests_done(seq_len_table) == False:
            rounds += 1
            if debug_on:
                print("--------------- new round ----------------")
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
            seg_latency = LayerNorm_latency(IA_rows, embedding_dim, die_num, N, mac_num, W, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, NonLinear_Statistics, D)
            latency_acc1(seg_latency, p_NonLinear_Statistics[0], D)
                    
            """ FC1 """
            # if debug_on:
            #     print("------------- FC1 -------------")
            # seg_latency = stage_latency(die_num, N, 3 * embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            # NOTE suppose at the very beginning of
            (seg_latency, nonlinear_latency) = stage_latency(die_num, N, 3 * embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, False, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, FC_Statistics, D)
            latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
            latency_substract(FC_Statistics, nonlinear_latency)
            # if debug_on:
            #     seg_latency.dump()
            #     Statistics.dump()
            
            """ Split """
            seg_latency = split_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, B, 
                                            3 * embedding_dim, IA_rows, mac_lane, mac_num, MH_alg_level_mode, J_max, J_min, MH_alg_distr_mode, args.sim_mode, debug_on)   
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, NonLinear_Statistics, D)
            latency_acc1(seg_latency, p_NonLinear_Statistics[1], D)
                    
            """ MH """
            # Scale is done immediately after MH1
            # if debug_on:
            #     print("------------- MH -------------")
            seg_latency = MH_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, B, H, 
                                        sram2_height, mac_lane, mac_num, head_embedding_dim, J_max, J_min, MH_alg_distr_mode, MH_alg_level_mode)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, MH_Statistics, D)
            # if debug_on:
            #     seg_latency.dump()
            #     Statistics.dump()
            
            """ Softmax """
            seg_latency = softmax_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, B, H, 
                                            head_embedding_dim, mac_lane, MH_alg_level_mode, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, NonLinear_Statistics, D)
            latency_acc1(seg_latency, p_NonLinear_Statistics[2], D)
            
            """ Concat """
            seg_latency = concat_latency(die_num, B, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, IA_rows, embedding_dim, mac_lane, mac_num, MH_alg_level_mode, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, NonLinear_Statistics, D)
            latency_acc1(seg_latency, p_NonLinear_Statistics[3], D)
            
            """ FC2 """
            # if debug_on:
            #     print("------------- FC2 -------------")
            # seg_latency = stage_latency(die_num, N, embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            (seg_latency, nonlinear_latency) = stage_latency(die_num, N, embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, True, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, FC_Statistics, D)
            latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
            latency_substract(FC_Statistics, nonlinear_latency)
            # if debug_on:
            #     seg_latency.dump()
            #     Statistics.dump()
            
            """ LN2 """
            seg_latency = LayerNorm_latency(IA_rows, embedding_dim, die_num, N, mac_num, W, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, NonLinear_Statistics, D)
            latency_acc1(seg_latency, p_NonLinear_Statistics[0], D)
            
            """ FC3 """
            # if debug_on:
            #     print("------------- FC3 -------------")
            # seg_latency = stage_latency(die_num, N, 4 * embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            (seg_latency, nonlinear_latency) = stage_latency(die_num, N, 4 * embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, False, debug_on)
            latency_acc1(seg_latency, Statistics, D)
            latency_acc1(seg_latency, FC_Statistics, D)
            latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
            latency_substract(FC_Statistics, nonlinear_latency)
            # if debug_on:
            #     seg_latency.dump()
            #     Statistics.dump()
            
            """ FC4 """
            # if debug_on:
            #     print("------------- FC4 -------------")
            # seg_latency = stage_latency(die_num, N, embedding_dim, 4 * embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
            (seg_latency, nonlinear_latency) = stage_latency(die_num, N, embedding_dim, 4 * embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, True, debug_on)
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
            for i in range(B):
                if seq_len_table[i][3] == False:
                    # if this is a live job
                    tmp_dynamic_weight += seq_len_table[i][2]
            tmp_dynamic_weight = 2 * embedding_dim * D * tmp_dynamic_weight
            Statistics.peak_dynamic_mem = max(tmp_dynamic_weight, Statistics.peak_dynamic_mem)
            
            debug_on = False
            flag = False
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
                            seq_len_table[i] = [True, requests_id, requests_pool[requests_id][1], False]
                            requests_pool[requests_id][0] = True
                            requests_id += 1         
                            # debug_on = True  
                            # debug_on_simple = True
                            flag = True
                else:
                    # in prefill stage, next time switch to generation stage
                    seq_len_table[i][0] = False
                    
            group_jobs = [0] * (die_num if MH_alg_level_mode == 0 else N)
            # for ii in range(B):
            #     if (seq_len_table[ii][3] == False) and (seq_len_table[ii][0] == False):
            #         group_jobs[die_idx_table[ii] if MH_alg_level_mode == 0 else core_idx_table[ii]] += 1
            # print(f"group_jobs after stopping: {group_jobs}")
            
        print(f"IA_rows_sum: {IA_rows_sum}")
        print(f"rounds: {rounds}")
        print(f"mh_count: {mh_count}")
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
                seq_len_table[i] = [True, i, max_prompt_len, False]
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
                seg_latency = LayerNorm_latency(IA_rows, embedding_dim, die_num, N, mac_num, W, debug_on)
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
                (seg_latency, nonlinear_latency) = stage_latency(die_num, N, 3 * embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, False, debug_on)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, FC_Statistics, D)
                latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
                latency_substract(FC_Statistics, nonlinear_latency)
                # if debug_on:
                #     seg_latency.dump()
                #     Statistics.dump()
                
                """ Split """
                seg_latency = split_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, b, 
                                                3 * embedding_dim, IA_rows, mac_lane, mac_num, MH_alg_level_mode, 0, 0, 0, args.sim_mode, debug_on=debug_on)   
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, NonLinear_Statistics, D)      
                latency_acc1(seg_latency, p_NonLinear_Statistics[1], D)
                
                """ MH """
                # latency += H * IA_latency # FIXME
                # if debug_on:
                #     print("------------- MH -------------")
                seg_latency =  MH_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, b, H, 
                                        sram2_height, mac_lane, mac_num, head_embedding_dim, 0, 0, 0, MH_alg_level_mode)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, MH_Statistics, D)
                # if debug_on:
                #     seg_latency.dump()
                #     Statistics.dump()
                
                """ Softmax """
                seg_latency = softmax_latency(die_num, N, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, b, H, 
                                                head_embedding_dim, mac_lane, MH_alg_level_mode, debug_on)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, NonLinear_Statistics, D)
                latency_acc1(seg_latency, p_NonLinear_Statistics[2], D)
                
                """ Concat """
                seg_latency = concat_latency(die_num, b, seq_len_table, die_idx_table if MH_alg_level_mode == 0 else core_idx_table, IA_rows, embedding_dim, mac_lane, mac_num, MH_alg_level_mode, debug_on)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, NonLinear_Statistics, D)
                latency_acc1(seg_latency, p_NonLinear_Statistics[3], D)
                
                """ FC2 """
                # latency += H * IA_latency # FIXME
                # if debug_on:
                #     print("------------- FC2 -------------")
                # seg_latency = stage_latency(die_num, N, embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
                (seg_latency, nonlinear_latency) = stage_latency(die_num, N, embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, True, debug_on)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, FC_Statistics, D)
                latency_acc1(nonlinear_latency, NonLinear_Statistics, D)
                latency_substract(FC_Statistics, nonlinear_latency)
                # if debug_on:
                #     seg_latency.dump()
                #     Statistics.dump()
                
                """ LN2 """
                seg_latency = LayerNorm_latency(IA_rows, embedding_dim, die_num, N, mac_num, W, debug_on)
                latency_acc1(seg_latency, Statistics, D)
                latency_acc1(seg_latency, NonLinear_Statistics, D)
                latency_acc1(seg_latency, p_NonLinear_Statistics[0], D)
                
                """ FC3 """
                # latency += H * IA_latency # FIXME
                # if debug_on:
                #     print("------------- FC3 -------------")
                # seg_latency = stage_latency(die_num, N, 4 * embedding_dim, embedding_dim, mac_lane, mac_num, 1, sram2_height, IA_rows, debug_on)
                (seg_latency, nonlinear_latency) = stage_latency(die_num, N, 4 * embedding_dim, embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, False, debug_on)
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
                (seg_latency, nonlinear_latency) = stage_latency(die_num, N, embedding_dim, 4 * embedding_dim, mac_lane, mac_num, W, sram2_height, IA_rows, p_NonLinear_Statistics, True, debug_on)
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
    
    return (total_latency, fc_latency, mh_latency, nonlinear_latency, nonlinear_latencys)
        
def main():
    """ Main function """

    args = argparser().parse_args()
    (requests_pool, s_avg) = init_requests_pool()
    print(f"average requests and answer length: {s_avg}")
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