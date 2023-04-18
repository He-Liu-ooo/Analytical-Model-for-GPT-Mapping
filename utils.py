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
        cal: latency of a mac_num dot production(8bit)
        psum_vec_cal: latency of adding 2 mac_lane * 3Byte data
        psum_cube_cal: latency of adding 2 mac_lane * mac_lane * 3Byte data
        psum_cube_cal: latency of adding N(core number in a die) mac_lane * mac_lane * 3Byte data
        psum_cube_loading: latency of reading a mac_lane * mac_lane * 3Byte data from HMC
        add_8: latency of adding 2 8-bit data together
        add_24: latency of adding 2 24-bit data together
        substract_square: latency of adding 2 (oprand1 - oprand2)^2, one oprand is 8-bit and the other is 24-bit
        division: latency of divide two data, one is 24-bit, one is 8-bit
        LN: latency of (x-EX)/(sqrt(Var+epsilon))*gamma + beta
        find_max: latency of finding maximum among mac_lane data
        substract_exp: latency of e^(x-x_max) of a data
        weight_loading: latency of transfering mac_num Byte of weight from HMC to SRAM of core or from SRAM of core to HMC
        IA_loading: latency of transfering mac_num Byte of IA from HMC to SRAM of core or from SRAM of core to HMC    
        psum_cube_wb: latency of writing a mac_lane * mac_lane * 3Byte data to HMC
        vec_wb: latency of writing a complete mac_lane Byte data to HMC
        vec_wb_sram: latency of writing a complete mac_lane Byte data to SRAM of core
        NoC_8: latency of transferring a 24-bit data of all cores to die's PPU or transferring data from die's PPU to all cores, propotion to the amount of data a core retains 
        NoC_24: latency of transferring a 8-bit data of all cores to die's PPU or transferring data from die's PPU to all cores, propotion to the amount of data a core retains 
        NoP_8: latency of transferring a 8-bit data from one die to another die or from a die to the shared PPU or from the shared PPU to all dies
        NoP_24: latency of transferring a 24-bit data from one die to another die or from a die to the shared PPU or from the shared PPU to all dies
        """
        
        self.cal = 0
        self.psum_vec_cal = 0
        self.psum_cube_cal = 0
        self.psum_cube_cal_N = 0
        self.psum_cube_loading = 0
        self.add_8 = 0
        self.add_24 = 0
        self.substract_square = 0
        self.division = 0
        self.LN = 0
        self.find_max = 0
        self.substract_exp = 0
        self.weight_loading = 0
        self.IA_loading = 0
        self.psum_cube_wb = 0
        self.vec_wb = 0
        self.vec_wb_sram = 0
        self.NoC_8 = 0
        self.NoC_24 = 0
        self.NoP_8 = 0
        self.NoP_24 = 0
    
    def multiple(self, N):
        self.cal *= N
        self.psum_vec_cal *= N
        self.psum_cube_cal *= N
        self.psum_cube_cal_N *= N
        self.psum_cube_loading *= N
        self.add_8 *= N
        self.add_24 *= N
        self.substract_square *= N
        self.division *= N
        self.LN *= N
        self.find_max *= N
        self.substract_exp *= N
        self.weight_loading *= N
        self.IA_loading *= N
        self.psum_cube_wb *= N
        self.vec_wb *= N
        self.vec_wb_sram *= N
        self.NoC_8 *= N
        self.NoC_24 *= N
        self.NoP_8 *= N
        self.NoP_24 *= N
        
    def get_params(self, cal_latency, psum_vec_cal_latency, psum_cube_cal_latency, psum_cube_cal_N_latency, psum_cube_loading_latency, add_8_latency, add_24_latency, substract_square_latency, division_latency, LN_latency,
                   find_max_latency, substract_exp_latency, weight_loading_latency, IA_loading_latency, psum_cube_wb_latency, vec_wb_latency, vec_wb_sram_latency, NoC_8_latency, NoC_24_latency, NoP_8_latency, NoP_24_latency):
        
        self.cal_latency = cal_latency
        self.psum_vec_cal_latency = psum_vec_cal_latency
        self.psum_cube_cal_latency = psum_cube_cal_latency
        self.psum_cube_cal_N_latency = psum_cube_cal_N_latency
        self.psum_cube_loading_latency = psum_cube_loading_latency
        self.add_8_latency = add_8_latency
        self.add_24_latency = add_24_latency
        self.substract_square_latency = substract_square_latency
        self.division_latency = division_latency
        self.LN_latency = LN_latency
        self.find_max_latency = find_max_latency
        self.substract_exp_latency = substract_exp_latency
        self.weight_loading_latency = weight_loading_latency
        self.IA_loading_latency = IA_loading_latency
        self.psum_cube_wb_latency = psum_cube_wb_latency
        self.vec_wb_latency = vec_wb_latency
        self.vec_wb_sram_latency = vec_wb_sram_latency
        self.NoC_8_latency = NoC_8_latency
        self.NoC_24_latency = NoC_24_latency
        self.NoP_8_latency = NoP_8_latency
        self.NoP_24_latency = NoP_24_latency
    
    def overall_latency(self):
        
        return (self.cal * self.cal_latency + self.psum_vec_cal * self.psum_vec_cal_latency + self.psum_cube_cal * self.psum_cube_cal_latency + self.psum_cube_cal_N * self.psum_cube_cal_N_latency +
                self.psum_cube_loading * self.psum_cube_loading_latency + self.add_8 * self.add_8_latency + self.add_24 * self.add_24_latency + self.substract_square * self.substract_square_latency +
                self.division * self.division_latency + self.LN * self.LN_latency + self.find_max * self. find_max_latency + self.substract_exp * self.substract_exp_latency + self.weight_loading * self.weight_loading_latency + self.IA_loading * self.IA_loading_latency +
                self.psum_cube_wb * self.psum_cube_wb_latency + self.vec_wb * self.vec_wb_latency + self.vec_wb_sram * self.vec_wb_sram_latency + self.NoC_8 * self.NoC_8_latency + self.NoC_24 * self.NoC_24_latency +
                self.NoP_8 * self.NoP_8_latency + self.NoP_24 * self.NoP_24_latency)
              
    def sum(self):
        return (self.cal + self.psum_vec_cal+ self.psum_cube_cal + self.psum_cube_cal_N + self.psum_cube_loading + self.add_8 + self.add_24 + self.substract_square + self.division +
                    self.LN + self.find_max + self.substract_exp + self.weight_loading + self.IA_loading + self.psum_cube_wb + self.vec_wb + self.vec_wb_sram + self.NoC_8 + self.NoC_24 + self.NoP_8 + self.NoP_24) 
    
    def dump(self):
        print("---------- dump latencys -----------")
        print(f"cal: {self.cal}")
        print(f"psum_vec_cal: {self.psum_vec_cal}")
        print(f"psum_cube_cal: {self.psum_cube_cal}")
        print(f"psum_cube_cal_N: {self.psum_cube_cal_N}")
        print(f"psum_cube_loading: {self.psum_cube_loading}")
        print(f"add_8: {self.add_8}")
        print(f"add_24: {self.add_24}")
        print(f"substract_square: {self.substract_square}")
        print(f"division: {self.division}")
        print(f"LN: {self.LN}")
        print(f"find_max: {self.find_max}")
        print(f"substract_exp: {self.substract_exp}")
        print(f"weight_loading: {self.weight_loading}")
        print(f"IA_loading: {self.IA_loading}")
        print(f"psum_cube_wb: {self.psum_cube_wb}")
        print(f"vec_wb: {self.vec_wb}")
        print(f"vec_wb_sram: {self.vec_wb_sram}")
        print(f"NoC_8: {self.NoC_8}")
        print(f"NoC_24: {self.NoC_24}")
        print(f"NoP_8: {self.NoP_8}")
        print(f"NoP_24: {self.NoP_24}")