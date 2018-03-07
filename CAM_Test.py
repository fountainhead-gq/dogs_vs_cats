base_model = VGG16(include_top=False, weights='imagenet')
for layers in base_model.layers:
    layers.trainable = False
y = GlobalAveragePooling2D()(base_model.output)
y = Dropout(0.25)(y)
y = Dense(1, activation='sigmoid')(y)
model = Model(inputs=base_model.input, outputs=y)
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_299, y_train_299, batch_size=32, epochs=1, validation_data=(X_valid_299, y_valid_299))

model2 = Model(inputs=base_model.input, outputs=[base_model.output, model.output])
out_base, out_final = model2.predict(np.expand_dims(X_train_299[1], axis=0))

weights = model.layers[-1].get_weights()[0]
print(out_base.shape, out_final.shape, weights.shape)



import random
img_width, img_height = 299, 299
plt.figure(figsize=(15, 16))
for i in range(1, 16):
    img = cv2.imread(os.path.join(path_test, '%d.jpg'% random.randint(1, X_test_299.shape[0])))
    img = cv2.resize(img,(img_width, img_height))
    x = img.copy()
    x.astype(np.float32)
    out, prediction = model2.predict(np.expand_dims(x, axis=0))
    prediction = prediction[0]
    out = out[0]    
    plt.subplot(3, 5, i)
    if prediction < 0.5:
        plt.title('cat %.2f%%' % (100 - prediction*100))
    else:
        plt.title('dog %.2f%%' % (prediction*100))
    
    cam = (prediction - 0.5) * np.matmul(out, weights)
    cam -= cam.min()
    cam /= cam.max()
    cam -= 0.2
    cam /= 0.8

    cam = cv2.resize(cam, (img_width, img_height))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam <= 0.2)] = 0

    out = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)

    plt.axis('off')
    plt.imshow(out[:,:,::-1])   