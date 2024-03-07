# def merge_numbers(original_list, desired_length):
#     merged_list = []
    
#     # 처음의 몇 개의 요소는 그대로 추가
#     for i in range(min(len(original_list), desired_length - 1)):
#         merged_list.append(original_list[i])
    
#     # 남은 요소들을 합쳐서 새로운 요소 하나로 만들어 추가
#     if len(original_list) > desired_length - 1:
#         merged_list.append(sum(original_list[desired_length - 1:]))
    
#     return merged_list

# # 예시 리스트와 원하는 길이 입력
# original_list = [1, 2, 3, 4]
# desired_length = 3

# # 함수 호출
# result = merge_numbers(original_list, desired_length)
# print(result)
#####################################
# def reduce_labels(labels, num_groups):
#     grouped_labels = []
#     labels_per_group = len(labels) // num_groups
#     remainder = len(labels) % num_groups
#     start_index = 0
#     for i in range(num_groups):
#         group_size = labels_per_group + (1 if i < remainder else 0)
#         end_index = start_index + group_size
#         grouped_labels.append((start_index, end_index - 1))
#         start_index = end_index
#     return grouped_labels

# # 예시 라벨과 원하는 그룹 개수 입력
# labels = list(range(7))  # 0부터 6까지의 숫자
# num_groups = 3

# # 함수 호출
# grouped_labels = reduce_labels(labels, num_groups)
# print(grouped_labels)