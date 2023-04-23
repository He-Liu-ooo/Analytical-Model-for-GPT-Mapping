import math

class Stats:
    def __init__(self):
        self.latency = latency()
        # unit: Byte
        self.peak_static_mem = 0
        self.peak_dynamic_mem = 0
        self.peak_IA = 0
        
    def dump(self, mode):
        print("------------ dump " + mode + " stats ------------")
        self.latency.dump()
        print(f"peak static memory usage: {self.peak_static_mem}")
        print(f"peak dynamic memory usage: {self.peak_dynamic_mem}")
        print(f"peak IA usage: {self.peak_IA}")
    
class latency:
    def __init__(self):
        """
        NOTE: all the latency unit is cycle
        
        cal: latency of a mac_num dot production(8bit)
        # psum_vec_cal: latency of adding 2 mac_lane * 3Byte data
        # psum_cube_cal: latency of adding 2 mac_lane * mac_lane * 3Byte data
        # psum_cube_cal: latency of adding N(core number in a die) mac_lane * mac_lane * 3Byte data
        # psum_cube_loading: latency of reading a mac_lane * mac_lane * 3Byte data from HMC
        
        ppu: latency of a PPU operation with data amount boundary
        # add_8: latency of adding 2 8-bit data together
        # add_24: latency of adding 2 24-bit data together
        # substract_square: latency of adding 2 (oprand1 - oprand2)^2, one oprand is 8-bit and the other is 24-bit
        # division: latency of divide two data, one is 24-bit, one is 8-bit
        # LN: latency of (x-EX)/(sqrt(Var+epsilon))*gamma + beta
        # find_max: latency of finding maximum among mac_lane data
        # substract_exp: latency of e^(x-x_max) of a data
        
        weight_loading: latency of transfering mac_num Byte of weight from HMC to SRAM of core or from SRAM of core to HMC
        IA_loading: latency of transfering mac_num Byte of IA from HMC to SRAM of core or from SRAM of core to HMC    
        # psum_cube_wb: latency of writing a mac_lane * mac_lane * 3Byte data to HMC
        # vec_wb: latency of writing a complete mac_lane Byte data to HMC
        # vec_wb_sram: latency of writing a complete mac_lane Byte data to SRAM of core
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
        self.IA_loading_latency = 0
        
        self.NoC_core_bandwidth = 0
        self.NoP_die_bandwidth = 0
        self.NoC_latency_per_hop = 0
        self.NoP_latency_per_hop = 0
        self.NoC_h = 0
        self.NoC_w = 0
        self.NoP_h = 0
        self.NoP_w = 0
        self.is_ring_mode = False
        self.NoC_latency = 0
        self.NoP_latency = 0
    
    def multiple(self, N):
        self.cal *= N
        self.ppu *= N
        self.weight_loading *= N
        self.dynamic_weight_loading *= N
        self.IA_loading *= N
        self.NoC_latency *= N
        self.NoP_latency *= N
        
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
    
    def overall_latency(self):
        
        self.calculate_latency()
        return (self.cal_latency + self.ppu_latency + self.weight_loading_latency + self.dynamic_weight_loading_latency + self.IA_loading_latency + self.NoC_latency + self.NoP_latency)
    
    def add_NoC_cores_to_ppu(self, data_amount):
        self.NoC_latency += (self.NoC_h - 1) * max(math.ceil(data_amount / self.NoC_core_bandwidth), 1 + self.NoC_latency_per_hop) \
                            + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop   
        # print(f"data_amount: {data_amount}")
        # print(f"NoC_cores_to_ppu: {(self.NoC_h - 1) * max(math.ceil(data_amount / self.NoC_core_bandwidth), 1 + self.NoC_latency_per_hop) + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop}")   
    
    def add_NoC_core_to_ppu(self, data_amount):
        n = 0
        for i in range(self.NoC_h):
            n += (i + 1)
        n = n / self.NoC_h
        
        self.NoC_latency += math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop * n          

    def add_NoC_ppu_to_cores_s(self, data_amount):
        if self.NoC_h == 1:
            self.NoC_latency += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
        elif self.NoC_h >= 2:
            self.NoC_latency += math.ceil(self.NoC_core_bandwidth / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop \
                                + (self.NoC_h - 2) * (1 + self.NoC_latency_per_hop) \
                                + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
        else:
            assert(0)
            
    def add_NoC_ppu_to_cores_m(self, data_amount):
        max_latency = 0
        if self.NoC_h >= 2:
            for i in range(self.NoC_h - 1):
                tmp_latency = i * math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) \
                                + math.ceil(self.NoC_core_bandwidth / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop \
                                + (self.NoC_h - 2 - i) * (1 + self.NoC_latency_per_hop) \
                                + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
                max_latency = max(tmp_latency, max_latency)
            self.NoC_latency += max_latency
        elif self.NoC_h == 1:
            self.NoC_latency += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
        else: 
            assert(0)
    
    def add_NoC_ppu_to_core(self, data_amount):
        """ ppu sends data to one core, in case of FC calculation """
        n = 0
        for i in range(self.NoC_h):
            n += (i + 1)
        n = n / self.NoC_h
         
        if math.ceil(n) == 1:
            self.NoC_latency += math.ceil(data_amount / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop
        elif n > 1:
            self.NoC_latency += math.ceil(self.NoC_core_bandwidth / (self.ppu_bandwidth / self.NoC_w)) + self.NoC_latency_per_hop \
                                + (self.NoC_h - 2 if self.NoC_h >= 2 else 0) * (1 + self.NoC_latency_per_hop) \
                                + math.ceil(data_amount / self.NoC_core_bandwidth) + self.NoC_latency_per_hop
        else:
            assert(0)
        
    def add_NoP_dies_to_ppu(self, data_amount):
        self.NoP_latency += (self.NoP_h - 1) * max(math.ceil(data_amount / self.NoP_die_bandwidth), 1 + self.NoP_latency_per_hop) \
                            + math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop          
        
    def add_NoP_ppu_to_dies_s(self, data_amount):
        if self.NoP_h == 1:
            self.NoP_latency += math.ceil(data_amount / (self.ppu_bandwidth / self.NoP_w)) + self.NoP_latency_per_hop
        elif self.NoP_h >= 2:
            self.NoP_latency += math.ceil(self.NoP_die_bandwidth / (self.ppu_bandwidth / self.NoP_w)) + self.NoP_latency_per_hop \
                                + (self.NoP_h - 2) * (1 + self.NoP_latency_per_hop) \
                                + math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
        else:
            assert(0)
            
    def add_NoP_rotation(self, data_amount):
        if self.is_ring_mode:
            self.NoP_latency += math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
        else:
            self.NoP_latency += math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop \
                                + (self.NoP_h + self.NoP_w - 2) * (1 + self.NoP_latency_per_hop) \
                                + math.ceil(data_amount / self.NoP_die_bandwidth) + self.NoP_latency_per_hop          
    
    def add_NoP_all_to_all(self, average_list):
        """ 
        average_list: a list contains average data amount that every core sents to other cores 
                      the order obeys row first 2D-1D
        """
        accum = 0
        for i in range(self.NoP_h):
            for j in range(self.NoP_w):
                accum += average_list[i * self.NoP_w + j] * max([abs(self.NoP_h - 1 - i) + abs(self.NoP_w - 1 - j), i + abs(self.NoP_w - 1 - j), abs(self.NoP_h - 1 - i) + j, i + j])
        
        self.NoP_latency += math.ceil(accum / self.NoP_die_bandwidth) + self.NoP_latency_per_hop
        
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
        print(f"NoC: {self.NoC_latency}")
        print(f"NoP: {self.NoP_latency}")
        
    def dump_portion(self, mode):
        print(f"---------- dump {mode} latency counts -----------")
        print(f"cal: {self.cal}")
        print(f"ppu: {self.ppu}")
        print(f"weight_loading: {self.weight_loading}")
        print(f"dynamic_weight_loading: {self.dynamic_weight_loading}")
        print(f"IA_loading: {self.IA_loading}")
        print(f"NoC_latency: {self.NoC_latency}")
        print(f"NoP_latency: {self.NoP_latency}")
        