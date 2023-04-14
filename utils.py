class Stats:
    def __init__(self):
        self.latency = latency()
        
    def dump(self, mode):
        print("------------ dump " + mode + " stats ------------")
        self.latency.dump()
    
class latency:
    def __init__(self):
        """
        cal: latency of a mac_num dot production(8bit)
        psum_vec_cal: latency of adding 2 mac_lane * 3Byte data
        psum_cube_cal: latency of adding 2 mac_lane * mac_lane * 3Byte data
        psum_cube_cal: latency of adding N(core number in a die) mac_lane * mac_lane * 3Byte data
        psum_cube_loading: latency of reading a mac_lane * mac_lane * 3Byte data from HMC
        weight_loading: latency of transfering mac_num Byte of weight from HMC to SRAM of core or from SRAM of core to HMC
        IA_loading: latency of transfering mac_num Byte of IA from HMC to SRAM of core or from SRAM of core to HMC    
        psum_cube_wb: latency of writing a mac_lane * mac_lane * 3Byte data to HMC
        vec_wb: latency of writing a complete mac_lane Byte data to HMC
        vec_wb_sram: latency of writing a complete mac_lane Byte data to SRAM of core
        NoP: latency of transferring a mac_num Byte data from one die to another die or from a die to the shared PPU
        """
        
        self.cal = 0
        self.psum_vec_cal = 0
        self.psum_cube_cal = 0
        self.psum_cube_cal_N = 0
        self.psum_cube_loading = 0
        self.weight_loading = 0
        self.IA_loading = 0
        self.psum_cube_wb = 0
        self.vec_wb = 0
        self.vec_wb_sram = 0
        self.NoP = 0
        
    def sum(self):
        return (self.cal + self.psum_vec_cal+ self.psum_cube_cal + self.psum_cube_cal_N + self.psum_cube_loading + self.weight_loading + self.IA_loading + self.psum_cube_wb + self.vec_wb + self.vec_wb_sram + self.NoP) 
    
    def dump(self):
        print("---------- dump latencys -----------")
        print(f"cal: {self.cal}")
        print(f"psum_vec_cal: {self.psum_vec_cal}")
        print(f"psum_cube_cal: {self.psum_cube_cal}")
        print(f"psum_cube_cal_N: {self.psum_cube_cal_N}")
        print(f"psum_cube_loading: {self.psum_cube_loading}")
        print(f"weight_loading: {self.weight_loading}")
        print(f"IA_loading: {self.IA_loading}")
        print(f"psum_cube_wb: {self.psum_cube_wb}")
        print(f"vec_wb: {self.vec_wb}")
        print(f"vec_wb_sram: {self.vec_wb_sram}")
        print(f"NoP: {self.NoP}")