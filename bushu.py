#
# import torch
# from torchvision import models
#
# net = models.resnet.resnet18(pretrained=True)
# dummpy_input = torch.randn(1,3,224,224)
# torch.onnx.export(net, dummpy_input, 'bast.onnx')
#
# import onnx
#
# # Load the ONNX model
# model = onnx.load(r'D:\lx\ultralytics-main\runs\detect\train19\weights\best.onnx')
#
# # Check that the IR is well formed
# onnx.checker.check_model(model)
#
# # Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))
#

import onnxruntime as rt
import numpy as  np
data = np.array(np.random.randn(1,3,224,224))
sess = rt.InferenceSession(r'D:\lx\ultralytics-main\runs\detect\train19\weights\best.onnx')
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onx = sess.run([label_name], {input_name:data.astype(np.float32)})[0]
print(pred_onx)
print(np.argmax(pred_onx))
