# Sample predictions for a random batch of images from the test set
samples = test.sample(15)
samples.reset_index(drop=True, inplace=True)

print("Generating captions for sample images...")

for index, record in samples.iterrows():
    img = load_img(os.path.join(image_path, record['image']), target_size=(224, 224))
    img = img_to_array(img) / 255.0
    
    caption = predict_caption(caption_model, record['image'], tokenizer, max_length, features)
    samples.loc[index, 'caption'] = caption

display_images(samples)
plt.show()