import cv2
import tensorflow as tf
import os
import numpy as np

# Enable GPU Memory Growth
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def contouringImg(image,mask):
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(mask[0].astype(np.uint8),kernel)
    dilation2 = cv2.dilate(mask[1].astype(np.uint8),kernel)
    edged = cv2.bitwise_and(dilation,dilation2)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0,255,0), thickness = 1)

    return image

# Metric Function
class MaxMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

# Loss Function
def dice_loss(y_true, y_pred, num_classes=2):
    smooth=tf.keras.backend.epsilon()
    dice=0
    for index in range(num_classes):
        y_true_f = tf.keras.backend.flatten(y_true[:,:,:,index])
        y_pred_f = tf.keras.backend.flatten(y_pred[:,:,:,index])
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
        dice += (intersection + smooth) / (union + smooth)
    return -2./num_classes * dice

# Parameter
image_size = (128, 128)
model_path = os.path.join("model", "model-unet.h5")
video_path = os.path.join("video.mp4")

# Video Writer
fps = 20 # An assumption
output_path = "output.mp4"
videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('I','4','2','0'),fps, image_size)

# Load model
model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss, 'MaxMeanIoU': MaxMeanIoU})


# Load Video
cap = cv2.VideoCapture(video_path)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        image_height = frame.shape[0]
        image_width = frame.shape[1]
        frame_ori = cv2.resize(frame,(500,500))
        frame = cv2.resize(frame, image_size)
        norm_frame = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

        # Predict mask
        pred = model.predict(np.expand_dims(norm_frame, 0))

        # Process mask
        mask = pred.squeeze()
        mask = np.stack((mask,)*3, axis=-1)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        mask_cutter = mask[:, :, 0]
        mask_bg = mask[:, :, 1]

        # Post Process
        mask_cutter = cv2.cvtColor(mask_cutter, cv2.COLOR_BGR2GRAY)
        mask_bg = cv2.cvtColor(mask_bg, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Mask",mask_cutter)
        cv2.imshow("Background",mask_bg)
        cv2.imshow("Original",frame_ori)
        cntImg = contouringImg(frame,(mask_cutter,mask_bg))
        cv2.imshow("Cutter Segmentation",cntImg)

        # Write frame
        videoWriter.write(cntImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

print("Video Ended")
cap.release()

cv2.destroyAllWindows()