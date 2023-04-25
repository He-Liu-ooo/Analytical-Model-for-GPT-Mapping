import math

class Stats:
    def __init__(self):
        self.latency = latency()
        # unit: Byte
        self.peak_static_mem = 0
        self.peak_dynamic_mem = 0
        self.peak_IA = 0
        self.peak_dynamic_mem_per_group = 0
        # average percentage that the peak group is above the average level
        # (peak-avg)/avg
        self.load_balance = 0
        
    def dump(self, mode, group_num=1):
        print("------------ dump " + mode + " stats ------------")
        self.latency.dump()
        print(f"peak static memory usage: {self.peak_static_mem}")
        print(f"peak dynamic memory usage: {self.peak_dynamic_mem}")
        print(f"peak IA usage: {self.peak_IA}")
        print(f"per group peak dynamic memory usage: {self.peak_dynamic_mem_per_group}")
        print(f"per group peak static memory usage: {self.peak_static_mem / group_num}")
        print(f"per group peak static+dynamic memory usage: {self.peak_static_mem / group_num + self.peak_dynamic_mem_per_group}")
        print(f"load balance: {self.load_balance}")
    
class latency:
    def __init__(self):
        """
        NOTE: all the latency unit is cycle
        
        cal: latency of a mac_num:1 dot production(8bit)
        
        ppu: latency of a PPU operation with data amount boundary
        
        weight_loading: latency of transfering mac_num Byte of weight from HMC to SRAM of core or from SRAM of core to HMC
        IA_loading: latency of transfering mac_num Byte of IA from HMC to SRAM of core or from SRAM of core to HMC    
        IA_loading_FC_operand: latency of loading IA matrix in FC
        IA_loading_FC_partialsum: latency of loading IA matrix in FC
        IA_loading_residual: latency of loading previous IA in residual connection
        IA_loading_FC_wb: latency of writing IA back to HMC in FC
        IA_loading_IA_rotation: latency of loading IA for rotation
        IA_loading_IA_rotation_wb: latency of writing IA back to HMC for rotation
        IA_loading_MH: latency of loading and writing back IA in MH
        IA_loading_softmax: latency of loading IA in softmax
        IA_loading_LN: latency of loading and write back IA in LN
        IA_loading_split_concat: latency of loading and write back IA for split head and concat
        
        NoC_latency_FC_partialsum: latency from FC adding results from all core together
        NoC_latency_IA_rotation: when IA rotation, data should be collected from all cores first then send to other dies
        NoC_latency_LN: NoC latency of LayerNorm
        NoC_latency_split_concat: NoC latency of split & concat, before sending data to other dies, we need to collect them from all cores
        NoC_latency_softmax: NoC latency of softmax
        NoP_latency_IA_rotation: latency of IA routing during IA rotation of FC
        NoP_latency_LN: NoP latency of LayerNorm
        NoP_latency_split_concat: NoP latency of split & concat
        NoP_latency_softmax: NoP latency of softmax
        
        NoC_transactions: number data need to be transferred in NoC, in the unit of BYTE
        NoP_transactions: number data need to be transferred in NoP, in the unit of BYTE
        """
        
        self.cal = 0
        self.p_cal = 0
        self.cal_latency = 0
        
        self.ppu = 0
        self.p_ppu = 0
        self.ppu_latency = 0
        self.ppu_boundary = 0
        self.ppu_bandwidth = 0
        
        self.weight_loading = 0
        self.p_weight_loading = 0
        self.weight_loading_latency = 0
        self.dynamic_weight_loading = 0
        self.dynamic_weight_loading_latency = 0
        
        self.p_IA_loading = 0
        self.IA_loading = 0
        self.IA_loading_FC_operand = 0
        self.IA_loading_FC_partialsum = 0
        self.IA_loading_residual = 0
        self.IA_loading_FC_wb = 0
        self.IA_loading_IA_rotation = 0
        self.IA_loading_IA_rotation_wb = 0
        self.IA_loading_MH = 0
        self.IA_loading_softmax = 0
        self.IA_loading_LN = 0
        self.IA_loading_split_concat = 0
        
        self.IA_loading_latency = 0
        self.IA_loading_FC_operand_latency = 0
        self.IA_loading_FC_partialsum_latency = 0
        self.IA_loading_residual_latency = 0
        self.IA_loading_FC_wb_latency = 0
        self.IA_loading_IA_rotation_latency = 0
        self.IA_loading_IA_rotation_wb_latency = 0
        self.IA_loading_MH_latency = 0
        self.IA_loading_softmax_latency = 0
        self.IA_loading_LN_latency = 0
        self.IA_loading_split_concat_latency = 0
        
        self.NoC_core_bandwidth = 0
        self.NoP_die_bandwidth = 0
        self.NoC_latency_per_hop = 0
        self.NoP_latency_per_hop = 0
        self.NoC_h = 0
        self.NoC_w = 0
        self.NoP_h = 0
        self.NoP_w = 0
        self.is_ring_mode = False
        # self.NoC_latency = 0
        # self.NoP_latency = 0
        # self.NoC_latency_FC_partialsum = 0
        # self.NoC_latency_IA_rotation = 0
        # self.NoC_latency_LN = 0
        # self.NoC_latency_split_concat = 0
        # self.NoC_latency_softmax = 0
        # self.NoP_latency_IA_rotation = 0
        # self.NoP_latency_LN = 0
        # self.NoP_latency_split_concat = 0
        # self.NoP_latency_softmax = 0
        
        self.NoC_transactions = 0
        self.NoC_transactions_FC_partialsum = 0
        self.NoC_transactions_IA_rotation = 0
        self.NoC_transactions_LN = 0
        self.NoC_transactions_split_concat = 0
        self.NoC_transactions_softmax = 0
        self.NoP_transactions = 0
        self.NoP_transactions_IA_rotation = 0
        self.NoP_transactions_LN = 0
        self.NoP_transactions_split_concat = 0
        self.NoP_transactions_softmax = 0
        
    
    def multiple(self, N):
        self.cal *= N
        self.ppu *= N
        self.weight_loading *= N
        self.dynamic_weight_loading *= N
        
        self.IA_loading *= N
        self.IA_loading_FC_operand *= N
        self.IA_loading_FC_partialsum *= N
        self.IA_loading_residual *= N
        self.IA_loading_FC_wb *= N
        self.IA_loading_IA_rotation *= N
        self.IA_loading_IA_rotation_wb *= N
        self.IA_loading_MH *= N
        self.IA_loading_softmax *= N
        self.IA_loading_LN *= N
        self.IA_loading_split_concat = N
        
        # self.NoC_latency *= N
        # self.NoP_latency *= N
        # self.NoC_latency_FC_partialsum *= N
        # self.NoC_latency_IA_rotation *= N
        # self.NoC_latency_LN *= N
        # self.NoC_latency_split_concat *= N
        # self.NoC_latency_softmax *= N
        self.NoC_transactions *= N
        self.NoP_transactions *= N
        self.NoC_transactions_FC_partialsum *= N
        self.NoC_transactions_IA_rotation *= N
        self.NoC_transactions_LN *= N
        self.NoC_transactions_split_concat *= N
        self.NoC_transactions_softmax *= N
        
        # self.NoP_latency_IA_rotation *= N
        # self.NoP_latency_LN *= N
        # self.NoP_latency_split_concat *= N
        # self.NoP_latency_softmax *= N
        self.NoP_transactions_IA_rotation *= N
        self.NoP_transactions_LN *= N
        self.NoP_transactions_split_concat *= N
        self.NoP_transactions_softmax *= N
        
    def get_params(self, params):
        
        self.p_cal = params[0]
        self.p_ppu = params[1]
        self.ppu_boundary = params[2]
        self.p_weight_loading = params[3]
        self.p_IA_loading = params[3]
        self.NoC_core_bandwidth = params[4]
        self.NoP_die_bandwidth = params[5]
        self.ppu_bandwidth = params[6]
        self.NoC_latency_per_hop = params[7]
        self.NoP_latency_per_hop = params[8]
        self.NoC_h = params[9]
        self.NoC_w = params[10]
        self.NoP_h = params[11]
        self.NoP_w = params[12]
        self.is_ring_mode = params[13]
    
    def calculate_latency(self):
        self.cal_latency = self.cal * self.p_cal
        self.ppu_latency = self.ppu * self.p_ppu
        self.weight_loading_latency = self.weight_loading * self.p_weight_loading
        self.dynamic_weight_loading_latency = self.dynamic_weight_loading * self.p_weight_loading
        
        self.IA_loading_latency = self.IA_loading * self.p_weight_loading
        self.IA_loading_FC_operand_latency = self.IA_loading_FC_operand * self.p_weight_loading
        self.IA_loading_FC_partialsum_latency = self.IA_loading_FC_partialsum * self.p_weight_loading
        self.IA_loading_residual_latency = self.IA_loading_residual * self.p_weight_loading
        self.IA_loading_FC_wb_latency = self.IA_loading_FC_wb * self.p_weight_loading
        self.IA_loading_IA_rotation_latency = self.IA_loading_IA_rotation * self.p_weight_loading
        self.IA_loading_IA_rotation_wb_latency = self.IA_loading_IA_rotation_wb * self.p_weight_loading
        self.IA_loading_MH_latency = self.IA_loading_MH * self.p_weight_loading
        self.IA_loading_softmax_latency = self.IA_loading_softmax * self.p_weight_loading
        self.IA_loading_LN_latency = self.IA_loading_LN * self.p_weight_loading
        self.IA_loading_split_concat_latency = self.IA_loading_split_concat * self.p_weight_loading
    
    def overall_latency(self):
        
        self.calculate_latency()
        # return (self.cal_latency + self.ppu_latency + self.weight_loading_latency + self.dynamic_weight_loading_latency + self.IA_loading_latency + self.NoC_latency + self.NoP_latency)
        # NOTE here NoC/NoP latency and ppu latency don't count
        return (self.cal_latency + self.weight_loading_latency + self.dynamic_weight_loading_latency + self.IA_loading_latency) # + self.NoC_latency + self.NoP_latency)
    
    """ inprecise NoC/NoP model """
    # def add_NoC_cores_to_ppu(self, data_amount, mode):
    #     self.NoC_latency += (self.NoC_h - 1) * max(math.ceil(data_amount / self.NoC_core_bandwidth), 1 + self.NoC_latency_per_hop) \
    #                         + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop   
    #     # print(f"data_amount: {data_amount}")
    #     # print(f"NoC_cores_to_ppu: {(self.NoC_h - 1) * max(math.ceil(data_amount / self.NoC_core_bandwidth), 1 + self.NoC_latency_per_hop) + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop}")   
    #     if mode == "FC_partialsum":
    #         self.NoC_latency_FC_partialsum += (self.NoC_h - 1) * max(math.ceil(data_amount / self.NoC_core_bandwidth), 1 + self.NoC_latency_per_hop) \
    #                         + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop  
    #     elif mode == "IA_rotation":
    #          self.NoC_latency_IA_rotation += (self.NoC_h - 1) * max(math.ceil(data_amount / self.NoC_core_bandwidth), 1 + self.NoC_latency_per_hop) \
    #                         + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop  
    #     elif mode == "LN":
    #         self.NoC_latency_LN += (self.NoC_h - 1) * max(math.ceil(data_amount / self.NoC_core_bandwidth), 1 + self.NoC_latency_per_hop) \
    #                         + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop  
    #     elif mode == "split_concat":
    #         self.NoC_latency_split_concat += (self.NoC_h - 1) * max(math.ceil(data_amount / self.NoC_core_bandwidth), 1 + self.NoC_latency_per_hop) \
    #                         + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
    #     elif mode == "softmax":
    #         self.NoC_latency_softmax += (self.NoC_h - 1) * max(math.ceil(data_amount / self.NoC_core_bandwidth), 1 + self.NoC_latency_per_hop) \
    #                         + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
    #     else:
    #         assert(0)
                            
    # def add_NoC_core_to_ppu(self, data_amount, mode):
    #     n = 0
    #     for i in range(self.NoC_h):
    #         n += (i + 1)
    #     n = n / self.NoC_h
        
    #     self.NoC_latency += math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop * n     
    #     if mode == "softmax":
    #         self.NoC_latency_softmax += math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop * n    
    #     else:
    #         assert(0)    

    # def add_NoC_ppu_to_cores_s(self, data_amount, mode):
    #     if self.NoC_h == 1:
    #         self.NoC_latency += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
    #         if mode == "LN":
    #             self.NoC_latency_LN += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
    #         elif mode == "softmax":
    #             self.NoC_latency_softmax += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
    #         else:
    #             assert(0)
    #     elif self.NoC_h >= 2:
    #         self.NoC_latency += math.ceil(self.NoC_core_bandwidth / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop \
    #                             + (self.NoC_h - 2) * (1 + self.NoC_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
    #         if mode == "LN":
    #             self.NoC_latency_LN += math.ceil(self.NoC_core_bandwidth / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop \
    #                             + (self.NoC_h - 2) * (1 + self.NoC_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
    #         elif mode == "softmax":
    #             self.NoC_latency_softmax += math.ceil(self.NoC_core_bandwidth / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop \
    #                             + (self.NoC_h - 2) * (1 + self.NoC_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
    #         else:
    #             assert(0)
    #     else:
    #         assert(0)
            
    # def add_NoC_ppu_to_cores_m(self, data_amount, mode):
    #     max_latency = 0
    #     if self.NoC_h >= 2:
    #         for i in range(self.NoC_h - 1):
    #             tmp_latency = i * math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) \
    #                             + math.ceil(self.NoC_core_bandwidth / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop \
    #                             + (self.NoC_h - 2 - i) * (1 + self.NoC_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
    #             max_latency = max(tmp_latency, max_latency)
    #         self.NoC_latency += max_latency
    #         if mode == "IA_rotation":
    #             self.NoC_latency_IA_rotation += max_latency
    #         elif mode == "split_concat":
    #             self.NoC_latency_split_concat += max_latency
    #         else:
    #             assert(0)
    #     elif self.NoC_h == 1:
    #         self.NoC_latency += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
    #         if mode == "IA_rotation":
    #             self.NoC_latency_IA_rotation += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
    #         elif mode == "split_concat":
    #             self.NoC_latency_split_concat += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
    #         else:
    #             assert(0)
    #     else: 
    #         assert(0)
    
    # def add_NoC_ppu_to_core(self, data_amount, mode):
    #     """ ppu sends data to one core, in case of FC calculation """
    #     n = 0
    #     for i in range(self.NoC_h):
    #         n += (i + 1)
    #     n = n / self.NoC_h
         
    #     if math.ceil(n) == 1:
    #         self.NoC_latency += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
    #         if mode == "FC_partialsum":
    #             self.NoC_latency_FC_partialsum += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
    #         elif mode == "softmax":
    #             self.NoC_latency_softmax += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
    #         else:
    #             assert(0)
    #     elif n > 1:
    #         self.NoC_latency += math.ceil(self.NoC_core_bandwidth / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop \
    #                             + (self.NoC_h - 2 if self.NoC_h >= 2 else 0) * (1 + self.NoC_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
    #         if mode == "FC_partialsum":
    #             self.NoC_latency_FC_partialsum += math.ceil(self.NoC_core_bandwidth / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop \
    #                             + (self.NoC_h - 2 if self.NoC_h >= 2 else 0) * (1 + self.NoC_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
    #         elif mode == "softmax":
    #             self.NoC_latency_softmax += math.ceil(self.NoC_core_bandwidth / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop \
    #                             + (self.NoC_h - 2 if self.NoC_h >= 2 else 0) * (1 + self.NoC_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
    #         else:
    #             assert(0)
    #     else:
    #         assert(0)
        
    # def add_NoP_dies_to_ppu(self, data_amount, mode):
    #     self.NoP_latency += (self.NoP_h - 1) * max(math.ceil(data_amount / self.NoP_die_bandwidth), 1 + self.NoP_latency_per_hop) \
    #                         + math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop          
    #     if mode == "LN":
    #         self.NoP_latency_LN += (self.NoP_h - 1) * max(math.ceil(data_amount / self.NoP_die_bandwidth), 1 + self.NoP_latency_per_hop) \
    #                         + math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
    #     elif mode == "softmax":
    #         self.NoP_latency_softmax += (self.NoP_h - 1) * max(math.ceil(data_amount / self.NoP_die_bandwidth), 1 + self.NoP_latency_per_hop) \
    #                         + math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
    #     else:
    #         assert(0)
            
    # def add_NoP_ppu_to_dies_s(self, data_amount, mode):
    #     if self.NoP_h == 1:
    #         self.NoP_latency += math.ceil(data_amount / (self.ppu_bandwidth / self.NoP_w)) + self.NoP_latency_per_hop
    #         if mode == "LN": 
    #             self.NoP_latency_LN += math.ceil(data_amount / (self.ppu_bandwidth / self.NoP_w)) + self.NoP_latency_per_hop
    #         elif mode == "softmax": 
    #             self.NoP_latency_softmax += math.ceil(data_amount / (self.ppu_bandwidth / self.NoP_w)) + self.NoP_latency_per_hop
    #         else:
    #             assert(0)
    #     elif self.NoP_h >= 2:
    #         self.NoP_latency += math.ceil(self.NoP_die_bandwidth / (self.ppu_bandwidth / self.NoP_w)) + self.NoP_latency_per_hop \
    #                             + (self.NoP_h - 2) * (1 + self.NoP_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
    #         if mode == "LN":
    #             self.NoP_latency_LN += math.ceil(self.NoP_die_bandwidth / (self.ppu_bandwidth / self.NoP_w)) + self.NoP_latency_per_hop \
    #                             + (self.NoP_h - 2) * (1 + self.NoP_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
    #         elif mode == "softmax":
    #             self.NoP_latency_softmax += math.ceil(self.NoP_die_bandwidth / (self.ppu_bandwidth / self.NoP_w)) + self.NoP_latency_per_hop \
    #                             + (self.NoP_h - 2) * (1 + self.NoP_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
    #         else:
    #             assert(0)
    #     else:
    #         assert(0)
            
    # def add_NoP_rotation(self, data_amount):
    #     if self.is_ring_mode:
    #         self.NoP_latency += math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
    #         self.NoP_latency_IA_rotation += math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
    #     else:
    #         self.NoP_latency += math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop \
    #                             + (self.NoP_h + self.NoP_w - 2) * (1 + self.NoP_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop      
    #         self.NoP_latency_IA_rotation += math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop \
    #                             + (self.NoP_h + self.NoP_w - 2) * (1 + self.NoP_latency_per_hop) \
    #                             + math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop          
    
    # def add_NoP_all_to_all(self, average_list):
        # """ 
        # average_list: a list contains average data amount that every core sents to other cores 
        #               the order obeys row first 2D-1D
        # """
        # accum = 0
        # for i in range(self.NoP_h):
        #     for j in range(self.NoP_w):
        #         accum += average_list[i * self.NoP_w + j] * max([abs(self.NoP_h - 1 - i) + abs(self.NoP_w - 1 - j), i + abs(self.NoP_w - 1 - j), abs(self.NoP_h - 1 - i) + j, i + j])
        
        # self.NoP_latency += math.ceil(accum / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
        # self.NoP_latency_split_concat += math.ceil(accum / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
        
    def add_ppu_count(self, data_amount):
        self.ppu += math.ceil(data_amount / self.ppu_boundary)
        
    def dump_params(self):
        print("---------- dump params -----------")
        print(f"cal_latency: {self.p_cal}")
        print(f"ppu_latency: {self.p_ppu}")
        print(f"ppu_boundary: {self.ppu_boundary}")
        print(f"weight/IA_loading_latency: {self.p_weight_loading}")
        print(f"NoC_core_bandwidth: {self.NoC_core_bandwidth}")
        print(f"NoP_die_bandwidth: {self.NoP_die_bandwidth}")
        print(f"ppu_bandwidth: {self.ppu_bandwidth}")
        print(f"NoC_latency_per_hop: {self.NoC_latency_per_hop}")
        print(f"NoP_latency_per_hop: {self.NoP_latency_per_hop}")
        print(f"NoC_h: {self.NoC_h}")
        print(f"NoC_w: {self.NoC_w}")
        print(f"NoP_h: {self.NoP_h}")
        print(f"NoP_w: {self.NoP_w}")
        print(f"is_ring_mode: {self.is_ring_mode}")
        
    def dump(self):
        print("---------- dump latencys -----------")
        print(f"cal: {self.cal_latency}")
        print(f"weight_loading: {self.weight_loading_latency}")
        print(f"dynamic_weight_loading: {self.dynamic_weight_loading_latency}")
        print(f"IA_loading: {self.IA_loading_latency}")
        print(f"ppu: {self.ppu_latency}")
        # print(f"NoC: {self.NoC_latency}")
        # print(f"NoP: {self.NoP_latency}")
        print(f"NoC: {self.NoC_transactions}")
        print(f"NoP: {self.NoP_transactions}")
        
    def dump_portion(self, mode):
        print(f"---------- dump {mode} latency counts -----------")
        print(f"cal: {self.cal}")
        print(f"ppu: {self.ppu}")
        print(f"weight_loading: {self.weight_loading}")
        print(f"dynamic_weight_loading: {self.dynamic_weight_loading}")
        print(f"IA_loading:                     {self.IA_loading}")
        if self.IA_loading != 0:
            print(f"+ IA_loading_FC_operand:     {self.IA_loading_FC_operand}-{round(100 * self.IA_loading_FC_operand / self.IA_loading, 2)}%")
            print(f"+ IA_loading_FC_partialsum:  {self.IA_loading_FC_partialsum}-{round(100 * self.IA_loading_FC_partialsum / self.IA_loading, 2)}%")
            print(f"+ IA_loading_residual:       {self.IA_loading_residual}-{round(100 * self.IA_loading_residual / self.IA_loading, 2)}%")
            print(f"+ IA_loading_FC_wb:          {self.IA_loading_FC_wb}-{round(100 * self.IA_loading_FC_wb / self.IA_loading, 2)}%")
            print(f"+ IA_loading_IA_rotation:    {self.IA_loading_IA_rotation}-{round(100 * self.IA_loading_IA_rotation / self.IA_loading, 2)}%")
            print(f"+ IA_loading_IA_rotation_wb: {self.IA_loading_IA_rotation_wb}-{round(100 * self.IA_loading_IA_rotation_wb / self.IA_loading, 2)}%")
            print(f"+ IA_loading_MH:             {self.IA_loading_MH}-{round(100 * self.IA_loading_MH / self.IA_loading, 2)}%")
            print(f"+ IA_loading_softmax:        {self.IA_loading_softmax}-{round(100 * self.IA_loading_softmax / self.IA_loading, 2)}%")
            print(f"+ IA_loading_LN:             {self.IA_loading_LN}-{round(100 * self.IA_loading_LN / self.IA_loading, 2)}%")
            print(f"+ IA_loading_split_concat:   {self.IA_loading_split_concat}-{round(100 * self.IA_loading_split_concat / self.IA_loading, 2)}%")
        else:
            print(f"+ IA_loading_FC_operand:     {self.IA_loading_FC_operand}")
            print(f"+ IA_loading_FC_partialsum:  {self.IA_loading_FC_partialsum}")
            print(f"+ IA_loading_residual:       {self.IA_loading_residual}")
            print(f"+ IA_loading_FC_wb:          {self.IA_loading_FC_wb}")
            print(f"+ IA_loading_IA_rotation:    {self.IA_loading_IA_rotation}")
            print(f"+ IA_loading_IA_rotation_wb: {self.IA_loading_IA_rotation_wb}")
            print(f"+ IA_loading_MH:             {self.IA_loading_MH}")
            print(f"+ IA_loading_softmax:        {self.IA_loading_softmax}")
            print(f"+ IA_loading_LN:             {self.IA_loading_LN}")
            print(f"+ IA_loading_split_concat:   {self.IA_loading_split_concat}")
            
        print(f"NoC_transactions:                     {self.NoC_transactions}")
        if self.NoC_transactions != 0:
            print(f"+ NoC_FC_partialsum:         {self.NoC_transactions_FC_partialsum}-{round(100 * self.NoC_transactions_FC_partialsum / self.NoC_transactions)}%")
            print(f"+ NoC_IA_rotation:           {self.NoC_transactions_IA_rotation}-{round(100 * self.NoC_transactions_IA_rotation / self.NoC_transactions)}%")
            print(f"+ NoC_LN:                    {self.NoC_transactions_LN}-{round(100 * self.NoC_transactions_LN / self.NoC_transactions)}%")
            print(f"+ NoC_split_concat:          {self.NoC_transactions_split_concat}-{round(100 * self.NoC_transactions_split_concat / self.NoC_transactions)}%")
            print(f"+ NoC_softmax:               {self.NoC_transactions_softmax}-{round(100 * self.NoC_transactions_softmax / self.NoC_transactions)}%")
        else:
            print(f"+ NoC_FC_partialsum:         {self.NoC_transactions_FC_partialsum}")
            print(f"+ NoC_IA_rotation:           {self.NoC_transactions_IA_rotation}")
            print(f"+ NoC_LN:                    {self.NoC_transactions_LN}")
            print(f"+ NoC_split_concat:          {self.NoC_transactions_split_concat}")
            print(f"+ NoC_softmax:               {self.NoC_transactions_softmax}")
            
        print(f"NoP_transactions:                     {self.NoP_transactions}")
        if self.NoP_transactions != 0:
            print(f"+ NoP_IA_rotation:           {self.NoP_transactions_IA_rotation}-{round(100 * self.NoP_transactions_IA_rotation / self.NoP_transactions)}%")
            print(f"+ NoP_split_concat:          {self.NoP_transactions_split_concat}-{round(100 * self.NoP_transactions_split_concat / self.NoP_transactions)}%")
            print(f"+ NoP_LN:                    {self.NoP_transactions_LN}-{round(100 * self.NoP_transactions_LN / self.NoP_transactions)}%")
            print(f"+ NoP_softmax:               {self.NoP_transactions_softmax}-{round(100 * self.NoP_transactions_softmax / self.NoP_transactions)}%")
        else:
            print(f"+ NoP_IA_rotation:           {self.NoP_transactions_IA_rotation}")
            print(f"+ NoP_split_concat:          {self.NoP_transactions_split_concat}")
            print(f"+ NoP_LN:                    {self.NoP_transactions_LN}")
            print(f"+ NoP_softmax:               {self.NoP_transactions_softmax}")
            
        # print(f"NoC_latency:                     {self.NoC_latency}")
        # if self.NoC_latency != 0:
        #     print(f"+ NoC_FC_partialsum:         {self.NoC_latency_FC_partialsum}-{round(100 * self.NoC_latency_FC_partialsum / self.NoC_latency)}%")
        #     print(f"+ NoC_IA_rotation:           {self.NoC_latency_IA_rotation}-{round(100 * self.NoC_latency_IA_rotation / self.NoC_latency)}%")
        #     print(f"+ NoC_LN:                    {self.NoC_latency_LN}-{round(100 * self.NoC_latency_LN / self.NoC_latency)}%")
        #     print(f"+ NoC_split_concat:          {self.NoC_latency_split_concat}-{round(100 * self.NoC_latency_split_concat / self.NoC_latency)}%")
        #     print(f"+ NoC_softmax:               {self.NoC_latency_softmax}-{round(100 * self.NoC_latency_softmax / self.NoC_latency)}%")
        # else:
        #     print(f"+ NoC_FC_partialsum:         {self.NoC_latency_FC_partialsum}")
        #     print(f"+ NoC_IA_rotation:           {self.NoC_latency_IA_rotation}")
        #     print(f"+ NoC_LN:                    {self.NoC_latency_LN}")
        #     print(f"+ NoC_split_concat:          {self.NoC_latency_split_concat}")
        #     print(f"+ NoC_softmax:               {self.NoC_latency_softmax}")
            
        # print(f"NoP_latency:                     {self.NoP_latency}")
        # if self.NoP_latency != 0:
        #     print(f"+ NoP_IA_rotation:           {self.NoP_latency_IA_rotation}-{round(100 * self.NoP_latency_IA_rotation / self.NoP_latency)}%")
        #     print(f"+ NoP_split_concat:          {self.NoP_latency_split_concat}-{round(100 * self.NoP_latency_split_concat / self.NoP_latency)}%")
        #     print(f"+ NoP_LN:                    {self.NoP_latency_LN}-{round(100 * self.NoP_latency_LN / self.NoP_latency)}%")
        #     print(f"+ NoP_softmax:               {self.NoP_latency_softmax}-{round(100 * self.NoP_latency_softmax / self.NoP_latency)}%")
        # else:
        #     print(f"+ NoP_IA_rotation:           {self.NoP_latency_IA_rotation}")
        #     print(f"+ NoP_split_concat:          {self.NoP_latency_split_concat}")
        #     print(f"+ NoP_LN:                    {self.NoP_latency_LN}")
        #     print(f"+ NoP_softmax:               {self.NoP_latency_softmax}")
        