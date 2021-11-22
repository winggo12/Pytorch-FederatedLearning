from fl_train import plot_partitions
from dataset_processing.fl_dataset import DatasetSplitByDirichletPartition

if __name__ == '__main__':
    user_num = 5
    alpha = 0.02
    file_path = '../BankChurners_normalized_standardized.csv'
    spliter = DatasetSplitByDirichletPartition(file_path=file_path,
                                               alpha=alpha,
                                               user_num=user_num,
                                               train_ratio=.7)
    dataset_dict = spliter.get_dataset_dict()
    proportions = spliter.get_train_dataset_proportions()
    label_partition_dict = spliter.get_label_partition_dict()

    spliter.save_users_df_as_csv(path="./")
    plot_partitions(user_num=user_num,
                    label_partition_dict=label_partition_dict,
                    alpha=alpha,
                    path="./")
    print("End")