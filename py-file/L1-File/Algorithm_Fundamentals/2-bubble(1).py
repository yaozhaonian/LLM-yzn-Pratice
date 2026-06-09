"""
双向冒泡排序（也叫鸡尾酒排序 / 搅拌排序）

双向遍历
左→右：把最大值冒泡到右侧未排序区间末尾
右→左：把最小值冒泡到左侧未排序区间开头

动态收缩左右边界，减少无效遍历
"""

def bubble_sort(arr):
    """
    改进版双向冒泡排序（鸡尾酒排序）
    每次遍历：左→右冒最大泡，右→左冒最小泡
    """
    n = len(arr)
    # 左边界：初始为0
    left = 0
    # 右边界：初始为最后一个元素索引
    right = n - 1
    
    while left < right:
        swapped = False
        
        # 1.左 -> 右: 把最大的元素冒泡到右边界
        for j in range(left, right):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        # 右边界左移一位(最大元素已就位)
        right -= 1
        
        # 如果没有交换，说明已有序，直接退出
        if not swapped:
            break
        
        swapped = False
        # 2.右 -> 左: 把最小的元素冒泡到左边界
        for j in range(right, left, -1):
            if arr[j] < arr[j-1]:
                arr[j], arr[j-1] = arr[j-1], arr[j]
                swapped = True
        # 左边界右移一位(最小元素已就位)
        left += 1
        
        # 如果没有交换，说明已有序，直接退出
        if not swapped:
            break
    
    return arr

# 测试数据与验证
test_data = [64, 34, 25, 12, 22, 11, 90]
print("排序前:", test_data)
bubble_sort(test_data)
print("排序后:", test_data)

"""
时间复杂度
    最坏时间复杂度
        O(n²)
        场景：数组完全逆序（如 [5,4,3,2,1]）
        原因：必须完整执行所有双向遍历，每一轮都要遍历几乎整个数组，总操作次数和传统冒泡排序一样，是 n × n 级别
    最好时间复杂度
        O(n)
        场景：数组已经有序（如 [1,2,3,4,5]）
        原因：
        只需要一次左→右遍历
        发现没有任何交换，直接退出
        总操作次数是线性的 n 次
        这一点和传统冒泡排序完全一样
    平均时间复杂度
        O(n²)
        随机乱序数组下，操作次数依然是平方级

空间复杂度：O(1)
属于原地排序
"""




