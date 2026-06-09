def bubble_sort(arr):
    """
    冒泡排序算法
    参数arr: 待排序的列表
    返回值: 排序后的列表
    """
    n = len(arr)
    # 外层循环: 控制总共需要几轮“冒泡”
    for i in range(n-1):
        # 假设本轮已经有序，用于优化
        swapped = False
        # 内层循环: 进行一轮冒泡，将最大的元素“沉”到最后
        # 每一轮后，最后 i+1 个元素是已排好的，所以范围是 n-1-i
        for j in range(0, n-1-i):
            # 如果前面的元素比后面的大，则交换
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        # 如果本轮一次交换都没发生，说明列表已完全有序，提前结束
        if not swapped:
            break
    return arr

# 测试数据与验证
test_data = [64, 34, 25, 12, 22, 11, 90]
print("排序前:", test_data)
sorted_data = bubble_sort(test_data.copy()) # 使用copy防止原列表被修改
print("排序后:", sorted_data)

# 验证结果是否正确
print("排序是否正确?", sorted_data == sorted(test_data.copy())) # 应与Python内置排序结果一致
    

"""
算法分析:
时间复杂度：
最坏与平均情况 O(n²)：当列表完全逆序时，需要进行 (n-1) + (n-2) + ... + 1 = n(n-1)/2 次比较和交换。
最好情况 O(n)：当列表已经有序时，加入 swapped 标志优化后，只需遍历一次即可发现无交换，提前结束。
空间复杂度 O(1)：算法只使用了 i, j, swapped 等固定数量的临时变量，是 原地排序 算法。
"""















