import openslide
from PIL import Image

class QueryFromWSI:
    def __init__(self, wsi_path, WH_ratio='4:3', MPixels=12, mpp=0.25, x_top_left=0, y_top_left=0):
        self.wsi_path = wsi_path
        self.WH_ratio = WH_ratio
        self.MPixels = MPixels
        self.mpp = mpp
        self.x_top_left = x_top_left
        self.y_top_left = y_top_left

        self.wsi = None
        self.query_image = None
        self.load_wsi()

    @property   
    def query_location(self):
        return self.x_top_left, self.y_top_left
    
    @property
    def query_image_height(self):
        '''
        The height of the query image MPixels
        '''
        spt_factor = self.MPixels * 1e6 / (int(self.WH_ratio.split(':')[0]) * int(self.WH_ratio.split(':')[1]))
        factor = spt_factor ** (1/2)
        return int(factor * int(self.WH_ratio.split(':')[1]))
    
    @property
    def query_image_width(self):
        spt_factor = self.MPixels * 1e6 / (int(self.WH_ratio.split(':')[0]) * int(self.WH_ratio.split(':')[1]))
        factor = spt_factor ** (1/2)
        return int(factor * int(self.WH_ratio.split(':')[0]))
    
    @property
    def query_FoV(self):
        return self.query_image_width * self.mpp, self.query_image_height * self.mpp
    
    
    def load_wsi(self):
        self.wsi = openslide.OpenSlide(self.wsi_path)

    def load_query_image(self):
        W_FoV, H_FoV = self.query_FoV
        
        # ================================================================
        # Get the mpp of the level 0 of the WSI
        # ================================================================
        props = self.wsi.properties
        mpp_x = None
        mpp_y = None    
        if "openslide.mpp-x" in props:
            mpp_x = float(props["openslide.mpp-x"])
        else:
            print("Warning: WSI mpp-x is not found")
        if "openslide.mpp-y" in props:
            mpp_y = float(props["openslide.mpp-y"])
        else:
            print("Warning: WSI mpp-y is not found")
        if mpp_x is None and mpp_y is None and "aperio.MPP" in props:
            mpp_x = mpp_y = float(props["aperio.MPP"])
            print("Warning: WSI mpp is not found, use aperio.MPP")
        if mpp_x is None or mpp_y is None:
            print("Warning: WSI mpp is not found")
            return None
        wsi_mpp = (mpp_x + mpp_y) / 2
        # ================================================================
        # Find the closest mpp level to the query mpp
        # if the query mpp close to the n-level wsi mpp, 
        #   use the n-level wsi mpp
        # else, 
        #   use the mpp of the level lower than the query mpp
        # ================================================================
        chosen_mpp = None
        chosen_level = None
        for n in range(len(self.wsi.level_downsamples)):
            mpp_level = self.wsi.level_downsamples[n] * wsi_mpp
            if abs(mpp_level - self.mpp) / mpp_level < 0.05:
                chosen_mpp = mpp_level
                chosen_level = n
                break
            elif mpp_level > self.mpp:
                chosen_mpp = self.wsi.level_downsamples[n-1] * wsi_mpp
                chosen_level = n-1
                break
        if chosen_mpp is None:
            print("Warning: No suitable mpp level found")
            return None
        # ================================================================
        # Get the query image from the WSI
        # ================================================================
        WSI_FoV_W = W_FoV / chosen_mpp
        WSI_FoV_H = H_FoV / chosen_mpp
        self.query_image = self.wsi.read_region((self.x_top_left, self.y_top_left), chosen_level, (int(WSI_FoV_W), int(WSI_FoV_H)))
        self.query_image = self.query_image.convert('RGB')
        self.query_image = self.query_image.resize((self.query_image_width, self.query_image_height), Image.LANCZOS)
        return self.query_image
    
    def change_query_location(self, x_top_left, y_top_left):
        '''
        Change the query location
        '''
        self.x_top_left = x_top_left
        self.y_top_left = y_top_left
        self.load_query_image()
        return self.query_image
    
    def change_query_MPixels(self, MPixels):
        '''
        Change the query MPixels
        '''
        self.MPixels = MPixels
        self.load_query_image()
        return self.query_image
    def change_query_mpp(self, mpp):
        '''
        Change the query mpp
        '''
        self.mpp = mpp
        self.load_query_image()
        return self.query_image
    def change_query_WH_ratio(self, WH_ratio):
        '''
        Change the query WH ratio
        '''
        self.WH_ratio = WH_ratio
        self.load_query_image()
        return self.query_image

