

class Translator():
    def __init__(self, shift, scale, drop):

        if shift != "no":
            ratio, direction = shift.split("-")
            ratio = float(ratio)
            shift = (ratio, direction)

        if scale != "no":
            ratio, direction = scale.split("-")
            ratio = float(ratio)
            scale = (ratio, direction)
            
        if drop != "no":
            param, criterion = drop.split("-")
            param = float(param)
            drop = (param, criterion)

        self.shift = shift
        self.scale = scale
        self.drop = drop
        
    def _scale_bbox(self, bbox, img_w, img_h):
        ratio, direction = self.scale
        assert ratio >= 0., "ratio need to be positive"
        assert direction in ["up", "down"], "direction need to be either up or down"
        
        x_top_left, y_top_left, bbox_width, bbox_height = bbox
        
        x_bottom_right = x_top_left + bbox_width
        y_bottom_right = y_top_left + bbox_height
        
        x_center = (x_top_left + x_bottom_right) / 2.
        y_center = (y_top_left + y_bottom_right) / 2.
        
        if direction == "up":
            y_top_left = y_center - bbox_height * (1+ratio) / 2.
            y_bottom_right = y_center + bbox_height * (1+ratio) / 2.
            
            x_top_left = x_center - bbox_width * (1+ratio) / 2.
            x_bottom_right = x_center + bbox_width * (1+ratio) / 2.
            
            y_top_left = max(0, y_top_left)
            y_bottom_right = min(img_h, y_bottom_right)
            
            x_top_left = max(0, x_top_left)
            x_bottom_right = min(img_w, x_bottom_right)
            
        if direction == "down":
            y_top_left = y_center - bbox_height * (1-ratio) / 2.
            y_bottom_right = y_center + bbox_height * (1-ratio) / 2.
            
            x_top_left = x_center - bbox_width * (1-ratio) / 2.
            x_bottom_right = x_center + bbox_width * (1-ratio) / 2.
            
            y_top_left = max(0, y_top_left)
            y_bottom_right = min(img_h, y_bottom_right)
            
            x_top_left = max(0, x_top_left)
            x_bottom_right = min(img_w, x_bottom_right)
            
        new_bbox_width = x_bottom_right - x_top_left
        new_bbox_height = y_bottom_right - y_top_left
        return [x_top_left, y_top_left, new_bbox_width, new_bbox_height]

    def _shift_bbox(self, bbox, img_w, img_h):
        ratio, direction = self.shift
        assert ratio >= 0., "ratio need to be positive"
        assert direction in ["left", "right", "up", "down"], "direction need to be either left, right, top, bottom"
        x_top_left, y_top_left, bbox_width, bbox_height = bbox
        
        x_bottom_right = x_top_left + bbox_width
        y_bottom_right = y_top_left + bbox_height
        
        if direction == "up":
            offset = ratio * bbox_height
            y_top_left = max(0, y_top_left - offset)
            y_bottom_right -= offset
            
        if direction == "down":
            offset = ratio * bbox_height
            y_top_left += offset
            y_bottom_right = min(img_h, y_bottom_right + offset)
            
        if direction == "left":
            offset = ratio * bbox_width
            
            x_top_left = max(0, x_top_left - offset)
            x_bottom_right -= offset
            
        if direction == "right":
            offset = ratio * bbox_width
            x_top_left += offset
            x_bottom_right = min(img_w, x_bottom_right + offset)
            
            
        new_bbox_height = y_bottom_right - y_top_left
        new_bbox_width = x_bottom_right - x_top_left
            
        return [x_top_left, y_top_left, new_bbox_width, new_bbox_height]

    def if_drop(self, bbox, **kwargs):
        if self.drop == "no":
            return False
        else:
            param, criterion=  self.drop
            if criterion == "small":
                bbox_size = kwargs["img_w"] * kwargs["img_h"]
                return bbox_size <= param
            elif criterion == "occluded":
                return kwargs["occluded"] > param
            elif criterion == "truncated":
                return kwargs["occluded"] > param
    
    def shift_or_scale(self, bbox, **kwargs):
        
        if self.shift != "no":
            bbox = self._shift_bbox(bbox, **kwargs)

        if self.scale != "no":
            bbox = self._scale_bbox(bbox, **kwargs)
            
        return bbox



