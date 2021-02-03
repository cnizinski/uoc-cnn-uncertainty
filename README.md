# uoc-cnn-uncertainty
Package for training keras CNNs on SEM images, making predictions with uncertainty, and visualizing learned features. For use with University of Utah uranium ore concentrate (UOC) images.

## Modules and Functions
### helpers.py
* img_info(fname, fields) - Extracts image information from regularly formatted SEM image filenames; provides rules for common naming discrepancies
* convert_fname(fname, fields) - Converts filenames to new standardized naming scheme
* quick_filter(df, filt_dict) - Filters pandas dataframes from dictionary where key is column name and value is list of columns values to filter for
* json2df(dpath, dfiles) - Reads multiple fson files from path into a single pandas dataframe
* split_dataset(dataframe, test_split, k, seed) - Train/Validation/Test splits
* stratified_split(df, label_col, test_split, balance, k, seed)- Stratified train/val./test splits
* oversample(df, label_col, balance) - Increases under-represented data classes in validation fold or train set
* drop_images(df, fname_col, img_path) - Checks that images in dataset can be loaded and removes bad images from set
* shannon_entropy(pred_list) - Computes Shannon information entropy from set of predicitions
### preprocessing.py
* random_crop(img, crop_size) - Square random crop of image
* pseudorandom_crop(img, crop_size, seedint) - Seeded square random crop
* center_crop(img, crop_size) - Square crop at center of image
* crop_generator(batches, crop_size, mode, seedint) - Custom crop function for Keras
* train_gen(tr_df, img_dir, num_classes, batch_size) - Keras image data generator for training/validation
* test_gen(vt_df, img_dir, num_classes, batch_size) - Keras image data generator for testing
### dropout.py
* dropout_layer(input_tensor, p=0.5, mc=False) - Returns Keras dropout layer usable for Monte Carlo dropout when mc=True
* get_...() - Returns Keras model of specified architecture
* unfreeze_all(model) - Makes all Keras model layers trainable
### train_test.py
* lr_decay(epoch, init_lr) - Decaying learning rate scheduler
* train_2steps(train_df, train_gen, model, params) - Training with base model weights frozen in p1 then unfrozen in p2
* mc_predict_df(test_df, img_path, label_idxs, model, n, crop) - MC Dropout inference on dataframe of testing data
### gradcam.py
* get_gradcam_v1(cropped_img, targets, model, last_conv, scale, series) - Returns GradCAM maps from cropped images
* normalize_resize_heatmaps(cam_dict, pred_series, thresh) - Resizes activation maps and normalizes to class probabilities
* single_gradcam(cropped_img, preds, scaled_cams, overlay) - Plots class activation maps for single image crop
* label_gradcam(corr_df, targets, true_label, model, img_path) - Plots CAMS for 25 image crops from same true label
* multiple_gradcam(corr_df, targets, model, img_path) - Plots CAMs for 5 crops from each class
