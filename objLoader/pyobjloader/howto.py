import tinyobjloader as tol
import json

model = tol.LoadObj("/u/leonliu/obj_samples/ff12c3a1d388b03044eedf822e07b7e4/models/model_normalized.obj")

#print(model["shapes"], model["materials"])
print( json.dumps(model, indent=4) )

#see cornell_box_output.json
