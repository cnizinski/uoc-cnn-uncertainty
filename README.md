# uoc_cnn_uncertainty
Package for training keras CNNs on SEM images, making predictions with uncertainty, and visualizing learned features. For use with University of Utah UOC SEM images.

## Modules and Functions
### helpers.py
Functions that help to view, clean, split, or extract data. Also includes functions for calculating information entropy.
* img_info(fname, fields) - Extracts image information from regularly formatted SEM image filenames; provides rules for common naming discrepancies
* convert_fname(fname, fields) - Converts filenames to new standardized naming scheme
* quick_filter(df, filt_dict) - Filters pandas dataframes from dictionary where key is column name and value is list of columns values to filter for
* json2df(dpath, dfiles) - Reads multiple fson files from path into a single pandas dataframe
* split_dataset(dataframe, test_split, k, seed) - Train/Validation/Test splits
* stratified_split(df, label_col, test_split, balance, k, seed)- Stratified train/val./test splits
* oversample(df, label_col, balance) - Increases under-represented data classes in validation fold or train set
* drop_images(df, fname_col, img_path) - Checks that images in dataset can be loaded and removes bad images from set
* shannon_entropy(pred_list) - Computes Shannon information entropy from set of predicitions
* kl_divergence(pred_dist, true_dist) - Computes Kullbeck-Leibler divergence from set of predictions and ground truth distribution
* series2list(pred_series, n_classes) - Encodes softmax score predictions for KL divergence calculations
* get_hfw(fname) - Returns image horizontal field width (HFW) from the filename
* get_scalebar(full_hfw, full_width, sub_width) - Returns size of scalebar for creating figures with SEM images
* convert_labels2sm(dpath, old_fname, new_fname, savefile) - Converts 16-class labels to 5-class for dataframe
### preprocessing.py
Preprocessing functions for CNN input images.
* random_crop(img, crop_size) - Square random crop of image
* pseudorandom_crop(img, crop_size, seedint) - Seeded square random crop
* center_crop(img, crop_size) - Square crop at center of image
* adaptive_crop1(img, crop_size, train_hfw, img_hfw) - Random crop that adjusts test image HFW to match the train image HFW
* adaptive_crop2(img, crop_size, train_hfw, img_hfw, seedint) - Seeded random crop that adjusts test HFW to train HFW
* crop_generator(batches, crop_size, mode, seedint) - Custom crop function for Keras
* train_gen(tr_df, img_dir, num_classes, batch_size) - Keras image data generator for training/validation
* test_gen(vt_df, img_dir, num_classes, batch_size) - Keras image data generator for testing
### models.py
Functions for creating keras CNN models.
* dropout_layer(input_tensor, p=0.5, mc=False) - Returns Keras dropout layer usable for Monte Carlo dropout when mc=True
* get_...() - Returns Keras model of specified architecture with specified parameters (see docstrings for details)
* unfreeze_all(model) - Makes all Keras model layers trainable
### train_test.py
Functions for tuning models and predicting on test sets with trained models.
* lr_decay(epoch, lr) - Decaying learning rate scheduler
* train_2steps(train_df, train_gen, model, params) - Training with base model weights frozen in p1 then unfrozen in p2
* mc_predict_image(test_gen, model, n) - MC dropout inference on single image (OBSOLETE and NOT THREADSAFE)
* mc_predict_df(test_df, img_path, label_idxs, model, n, crop) - MC Dropout inference on dataframe of testing data
### visualize.py
Functions for visualizing learned features with GradCAM (needs work) or making confusion matrix figures
* get_gradcam_v1(cropped_img, targets, model, last_conv, scale, series) - Returns GradCAM maps from cropped images
* normalize_resize_gradcam(cam_dict, pred_series, thresh) - Resizes activation maps and normalizes to class probabilities
* single_image_plot(cropped_img, preds, scaled_disct, overlay) - Creates class activation map figure for single image (only works for 5-class predictions)
* single_class_plot(corr_df, targets, true_label, model, img_path, conf) - Creates figure with CAMs for single class with minimum confidence score
* multi_class_plot(corr_df, targets, model, img_path, conf) - Creates figure with CAMs for single class with minimum confidence score
* triple_cm(df_5u, df_5f, df_16, title, report) - Creates confusion matrix figure for 2 sets of 5-class predictions and 1 set of 16-class predicitons
