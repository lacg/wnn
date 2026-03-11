  Standard Split (175K train / 82K test — what we use)                                                                                                                            
                                                                                                                                                                                  
  This is the harder, more realistic split. Results are significantly lower:                                                                                                      
                                                                                                                                                                                  
  ┌────────────────────────┬──────────┬─────────┬─────────┐
  │         Model          │ Accuracy │   F1    │   FPR   │
  ├────────────────────────┼──────────┼─────────┼─────────┤
  │ Random Forest          │ ~87-88%  │ ~87-88% │ ~10-15% │
  ├────────────────────────┼──────────┼─────────┼─────────┤
  │ XGBoost                │ ~87%     │ ~87%    │ ~12%    │
  ├────────────────────────┼──────────┼─────────┼─────────┤
  │ SVM                    │ ~86%     │ ~85%    │ ~15%    │
  ├────────────────────────┼──────────┼─────────┼─────────┤
  │ DNN                    │ ~88%     │ ~87%    │ ~10%    │
  ├────────────────────────┼──────────┼─────────┼─────────┤
  │ BiLSTM (deep learning) │ ~89%     │ ~89%    │ ~8%     │
  └────────────────────────┴──────────┴─────────┴─────────┘

  Random Split (70/30 or 80/20 random shuffle)

  Much easier — inflated numbers due to data leakage between train/test:

  ┌───────────────┬──────────┬────────┬─────┐
  │     Model     │ Accuracy │   F1   │ FPR │
  ├───────────────┼──────────┼────────┼─────┤
  │ RF / XGBoost  │ 98-99%   │ 97-99% │ <2% │
  ├───────────────┼──────────┼────────┼─────┤
  │ Deep Learning │ 98-99%   │ 98-99% │ <1% │
  └───────────────┴──────────┴────────┴─────┘

  Realistic Goals for Our WNN

  On the standard split (which we're using), reasonable targets:

  ┌──────────┬───────────────────────┬────────────────────┬────────────────┐
  │  Metric  │ Classical ML baseline │ Deep Learning SOTA │ Our WNN target │
  ├──────────┼───────────────────────┼────────────────────┼────────────────┤
  │ F1-Macro │ ~87%                  │ ~89%               │ 85-88%         │
  ├──────────┼───────────────────────┼────────────────────┼────────────────┤
  │ FPR      │ ~10-15%               │ ~8%                │ <12%           │
  ├──────────┼───────────────────────┼────────────────────┼────────────────┤
  │ Accuracy │ ~87%                  │ ~89%               │ 85-88%         │
  └──────────┴───────────────────────┴────────────────────┴────────────────┘

  Matching the RF baseline (~87% F1, ~12% FPR) would be a strong result for a weightless neural network. Beating it would be publishable. The standard split is hard because the
  train/test distributions differ significantly (the dataset authors deliberately made it challenging).

  Where is experiment 1576 right now on these metrics?

  Sources:
  - Zoghi 2024 - Building an IDS on UNSW-NB15
  - Enhanced IDS Performance with UNSW-NB15
  - Performance Analysis using Feature Selection on UNSW-NB15
  - UNSW-NB15 Dataset
  - Oversampling, Stacking Feature Embedding (2024)