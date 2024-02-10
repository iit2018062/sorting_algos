import random
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import sys

# Set recursion limit
sys.setrecursionlimit(10 ** 6)
MIN_MERGE = 32


# Quick sort algorithm
class QuickSort:
    def partition(self, nums, start, end):
        i = start - 1
        pivot = nums[end]
        for j in range(start, end):
            if nums[j] <= pivot:
                i += 1
                (nums[i], nums[j]) = (nums[j], nums[i])
        (nums[i + 1], nums[end]) = (nums[end], nums[i + 1])
        return i + 1

    def algorithm(self, nums, start, end):
        if start < end:
            pivot = self.partition(nums, start, end)
            self.algorithm(nums, start, pivot - 1)
            self.algorithm(nums, pivot + 1, end)


# Heap sort algorithm
class HeapSort:
    def algorithm(self, array):
        n = len(array)
        i = n // 2 - 1
        while i >= 0:
            self.heapify(array, n, i)
            i -= 1
        i = n - 1
        while i > 0:
            array[i], array[0] = array[0], array[i]
            self.heapify(array, i, 0)
            i -= 1

    def heapify(self, nums, n, i):
        left = 2 * i + 1
        largest = i
        right = 2 * i + 2
        if left < n and nums[left] > nums[largest]:
            largest = left
        if right < n and nums[right] > nums[largest]:
            largest = right
        if largest != i:
            nums[i], nums[largest] = nums[largest], nums[i]
            self.heapify(nums, n, largest)


class BucketSort():
    def algorithm(self, nums):
        if not any(nums):
            return nums
        # finding the min and max in the array
        min_value = min(nums)
        max_value = max(nums)
        # finding the range of values to get bucket size
        range_value = max_value - min_value
        size = range_value / len(nums) if len(nums) != 0 else 1
        buckets = []
        for _ in range(len(nums)):
            buckets.append([])
        # assigning values to buckets
        for i in range(len(nums)):
            if size == 0:
                j = 0
            else:
                j = int((nums[i] - min_value) / size)
            if j != len(nums):
                buckets[j].append(nums[i])
            else:
                buckets[len(nums) - 1].append(nums[i])
        for k in range(len(nums)):
            self.insertionSort(buckets[k])
        result = []
        for x in range(len(nums)):
            result.extend(buckets[x])
        return result

    def insertionSort(self, bucket_tmp):
        for i in range(1, len(bucket_tmp)):
            v = bucket_tmp[i]
            for j in range(i - 1, -1, -1):
                if v < bucket_tmp[j]:
                    bucket_tmp[j + 1] = bucket_tmp[j]
                else:
                    break
            bucket_tmp[j + 1] = v


# Merge sort algorithm
class MergeSort:
    def algorithm(self, nums=[]):
        if len(nums) < 2:
            return nums
        mid = len(nums) // 2
        left = self.algorithm(nums[:mid]) if mid > 0 else []
        right = self.algorithm(nums[mid:])
        return self.merge_array(nums, left, right)

    def merge_array(self, array, left, right):
        result = []
        i, j = 0, 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        array[:] = result
        return result


# Radix sort algorithm
class RadixSort:
    def algorithm(self, array):
        def counting_sort(array, exp):
            output = [0] * len(array)
            count = [0] * (10)
            for i in range(0, len(array)):
                idx = (array[i] // exp)
                count[int((idx) % 10)] += 1
            for i in range(1, 10):
                count[i] += count[i - 1]
            i = len(array) - 1
            while i >= 0:
                idx = (array[i] // exp)
                output[count[int((idx) % 10)] - 1] = array[i]
                count[int((idx) % 10)] -= 1
                i -= 1
            i = 0
            for i in range(len(array)):
                array[i] = output[i]

        maximum = max(array)
        exp = 1
        while maximum // exp > 0:
            counting_sort(array, exp)
            exp *= 10


# Tim sort algorithm

class TimSort:
    MIN_MERGE = 32

    def calc_min_run(self, n):
        r = 0
        while n >= self.MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r

    def insertion_sort(self, arr, left, right):
        for i in range(left + 1, right + 1):
            j = i
            while j > left and arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                j -= 1

    def merge(self, arr, l, m, r):
        len1, len2 = m - l + 1, r - m
        left, right = [], []
        for i in range(0, len1):
            left.append(arr[l + i])
        for i in range(0, len2):
            right.append(arr[m + 1 + i])

        i, j, k = 0, 0, l

        while i < len1 and j < len2:
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len1:
            arr[k] = left[i]
            k += 1
            i += 1

        while j < len2:
            arr[k] = right[j]
            k += 1
            j += 1

    def tim_sort(self, arr):
        n = len(arr)
        min_run = self.calc_min_run(n)

        for start in range(0, n, min_run):
            end = min(start + min_run - 1, n - 1)
            self.insertion_sort(arr, start, end)

        size = min_run
        while size < n:
            for left in range(0, n, 2 * size):
                mid = min(n - 1, left + size - 1)
                right = min((left + 2 * size - 1), (n - 1))

                if mid < right:
                    self.merge(arr, left, mid, right)

            size = 2 * size


def scenario_1(size):
    return randint(0, size, size=size)


# [0,k], k<1000
def scenario_2(size):
    return randint(0, 999, size=size)


# [0,n^3]
def scenario_3(size):
    return randint(0, size ** 3, size=size)


# [0,log(n)]
def scenario_4(size):
    return randint(0, int(np.log(size)), size=size)


# [0,n](*1000)
def scenario_5(size):
    nums = []
    for _ in range(size):
        nums.append(random.randint(0, size // 1000) * 1000)

    return nums


# [0,n](swapped)
def scenario_6(size):
    number_swaps = int((np.log(size) / 2) // 1)
    sequence = list(range(size + 1))
    for i in range(number_swaps):
        index1, index2 = random.sample(range(size + 1), 2)
        sequence[index1], sequence[index2] = sequence[index2], sequence[index1]
    return sequence


def graph_quick_sort(times_quick,elements):
    for scenario, times in times_quick.items():
        plt.plot(elements, times, label=f'QuickSort - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()


def graph_heap_sort(times_heap,elements):
    for scenario, times in times_heap.items():
        plt.plot(elements, times, label=f'heapsort - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()


def graph_merge_sort(times_merge,elements):
    for scenario, times in times_merge.items():
        plt.plot(elements, times, label=f'mergeSort - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()


def graph_radix_sort(times_radix,elements):
    for scenario, times in times_radix.items():
        plt.plot(elements, times, label=f'radixSort - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()


def graph_bucket_sort(times_bucket,elements):
    for scenario, times in times_bucket.items():
        plt.plot(elements, times, label=f'bucketSort - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()


def graph_tim_sort(times_tim_sort,elements):
    for scenario, times in times_tim_sort.items():
        plt.plot(elements, times, label=f'timSort - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()



def graph_scenario_1(times_scenario_5,elements):
    for scenario, times in times_scenario_5.items():
        if len(elements) == len(times):
            plt.plot(elements, times, label=f'algo - {scenario}')
        else:
            print(f"Skipping {scenario} due to mismatched dimensions.")

    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()
def graph_scenario_2(times_scenario_2, elements ):
    for scenario, times in times_scenario_2.items():
        plt.plot(elements, times, label=f'algo - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()

def graph_scenario_3(times_scenario_3, elements):
    for scenario, times in times_scenario_3.items():
        plt.plot(elements, times, label=f'aglo - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()

def graph_scenario_4(times_scenario_4, elements):
    for algo, times in times_scenario_4.items():
        plt.plot(elements, times, label=f'algo - {algo}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()

def graph_scenario_5(times_scenario_5, elements):
    for algo, times in times_scenario_5.items():
        plt.plot(elements, times, label=f'algo - {algo}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()

def graph_scenario_6(times_scenario_6, elements):
    for algo, times in times_scenario_6.items():
        plt.plot(elements, times, label=f'algo - {algo}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()

# Initialize QuickSort and HeapSort objects
quick = QuickSort()
heap = HeapSort()
merge = MergeSort()
radix = RadixSort()
tim = TimSort()
bucket = BucketSort()


def senario_vs_sort_algorithm():
    time_scenario_1 = {"quick": [], "heap": [], "merge": [], "radix": [], "bucket": [],
                       "tim": []}
    time_scenario_2 = {"quick": [], "heap": [], "merge": [], "radix": [], "bucket": [],
                       "tim": []}
    time_scenario_3 = {"quick": [], "heap": [], "merge": [], "radix": [], "bucket": [],
                       "tim": []}
    time_scenario_4 = {"quick": [], "heap": [], "merge": [], "radix": [], "bucket": [],
                       "tim": []}
    time_scenario_5 = {"quick": [], "heap": [], "merge": [], "radix": [], "bucket": [],
                       "tim": []}
    time_scenario_6 = {"quick": [], "heap": [], "merge": [], "radix": [], "bucket": [],
                       "tim": []}
    elements = []
    for i in range(1000, 10000, 1000):
        nums_scenario_1 = scenario_1(i)
        elements.append(len(nums_scenario_1))
        nums_quick = nums_scenario_1
        start_quick = time.process_time()
        quick.algorithm(nums_quick, 0, len(nums_quick) - 1)
        end_quick = time.process_time()
        time_scenario_1["quick"].append(end_quick - start_quick)
        nums_merge = nums_scenario_1
        start_merge = time.process_time()
        merge.algorithm(nums_merge)
        end_merge = time.process_time()
        time_scenario_1["merge"].append(end_merge - start_merge)
        nums_heap = nums_scenario_1
        start_heap = time.process_time()
        heap.algorithm(nums_heap)
        end_heap = time.process_time()
        time_scenario_1["heap"].append(end_heap - start_heap)
        nums_radix = nums_scenario_1
        start_radix = time.process_time()
        radix.algorithm(nums_radix)
        end_radix = time.process_time()
        time_scenario_1["radix"].append(end_radix - start_radix)
        nums_bucket = nums_scenario_1
        start_bucket = time.process_time()
        bucket.algorithm(nums_bucket)
        end_bucket = time.process_time()
        time_scenario_1["bucket"].append(end_bucket - start_bucket)
        nums_tim = nums_scenario_1
        start_tim = time.process_time()
        tim.tim_sort(nums_tim)
        end_tim = time.process_time()
        time_scenario_1["tim"].append(end_tim - start_tim)
    elements=[]
    for i in range(1000, 10000, 1000):
        nums_scenario_2 = scenario_2(i)
        elements.append(len(nums_scenario_2))
        nums_quick = nums_scenario_2
        start_quick = time.process_time()
        quick.algorithm(nums_quick, 0, len(nums_quick) - 1)
        end_quick = time.process_time()
        time_scenario_2["quick"].append(end_quick - start_quick)
        nums_merge = nums_scenario_2
        start_merge = time.process_time()
        merge.algorithm(nums_merge)
        end_merge = time.process_time()
        time_scenario_2["merge"].append(end_merge - start_merge)
        nums_heap = nums_scenario_2
        start_heap = time.process_time()
        heap.algorithm(nums_heap)
        end_heap = time.process_time()
        time_scenario_2["heap"].append(end_heap - start_heap)
        nums_radix = nums_scenario_2
        start_radix = time.process_time()
        radix.algorithm(nums_radix)
        end_radix = time.process_time()
        time_scenario_2["radix"].append(end_radix - start_radix)
        nums_bucket = nums_scenario_2
        start_bucket = time.process_time()
        bucket.algorithm(nums_bucket)
        end_bucket = time.process_time()
        time_scenario_2["bucket"].append(end_bucket - start_bucket)
        nums_tim = nums_scenario_2
        start_tim = time.process_time()
        tim.tim_sort(nums_tim)
        end_tim = time.process_time()
        time_scenario_2["tim"].append(end_tim - start_tim)
    elements=[]
    for i in range(1000, 10000, 1000):
        nums_scenario_3 = scenario_3(i)
        elements.append(len(nums_scenario_3))
        nums_quick = nums_scenario_3
        start_quick = time.process_time()
        quick.algorithm(nums_quick, 0, len(nums_quick) - 1)
        end_quick = time.process_time()
        time_scenario_3["quick"].append(end_quick - start_quick)
        nums_merge = nums_scenario_3
        start_merge = time.process_time()
        merge.algorithm(nums_merge)
        end_merge = time.process_time()
        time_scenario_3["merge"].append(end_merge - start_merge)
        nums_heap = nums_scenario_3
        start_heap = time.process_time()
        heap.algorithm(nums_heap)
        end_heap = time.process_time()
        time_scenario_3["heap"].append(end_heap - start_heap)
        nums_radix = nums_scenario_3
        start_radix = time.process_time()
        radix.algorithm(nums_radix)
        end_radix = time.process_time()
        time_scenario_3["radix"].append(end_radix - start_radix)
        nums_bucket = nums_scenario_3
        start_bucket = time.process_time()
        bucket.algorithm(nums_bucket)
        end_bucket = time.process_time()
        time_scenario_3["bucket"].append(end_bucket - start_bucket)
        nums_tim = nums_scenario_3
        start_tim = time.process_time()
        tim.tim_sort(nums_tim)
        end_tim = time.process_time()
        time_scenario_3["tim"].append(end_tim - start_tim)
    elements=[]
    for i in range(1000, 10000, 1000):
        nums_scenario_4 = scenario_4(i)
        elements.append(len(nums_scenario_4))
        nums_quick = nums_scenario_4
        start_quick = time.process_time()
        quick.algorithm(nums_quick, 0, len(nums_quick) - 1)
        end_quick = time.process_time()
        time_scenario_4["quick"].append(end_quick - start_quick)
        nums_merge = nums_scenario_4
        start_merge = time.process_time()
        merge.algorithm(nums_merge)
        end_merge = time.process_time()
        time_scenario_4["merge"].append(end_merge - start_merge)
        nums_heap = nums_scenario_4
        start_heap = time.process_time()
        heap.algorithm(nums_heap)
        end_heap = time.process_time()
        time_scenario_4["heap"].append(end_heap - start_heap)
        nums_radix = nums_scenario_4
        start_radix = time.process_time()
        radix.algorithm(nums_radix)
        end_radix = time.process_time()
        time_scenario_4["radix"].append(end_radix - start_radix)
        nums_bucket = nums_scenario_4
        start_bucket = time.process_time()
        bucket.algorithm(nums_bucket)
        end_bucket = time.process_time()
        time_scenario_4["bucket"].append(end_bucket - start_bucket)
        nums_tim = nums_scenario_4
        start_tim = time.process_time()
        tim.tim_sort(nums_tim)
        end_tim = time.process_time()
        time_scenario_4["tim"].append(end_tim - start_tim)
    elements=[]
    for i in range(1000, 10000, 1000):
        nums_scenario_5 = scenario_5(i)
        elements.append(len(nums_scenario_5))
        nums_quick = nums_scenario_5
        start_quick = time.process_time()
        quick.algorithm(nums_quick, 0, len(nums_quick) - 1)
        end_quick = time.process_time()
        time_scenario_5["quick"].append(end_quick - start_quick)
        nums_merge = nums_scenario_5
        start_merge = time.process_time()
        merge.algorithm(nums_merge)
        end_merge = time.process_time()
        time_scenario_5["merge"].append(end_merge - start_merge)
        nums_heap = nums_scenario_5
        start_heap = time.process_time()
        heap.algorithm(nums_heap)
        end_heap = time.process_time()
        time_scenario_5["heap"].append(end_heap - start_heap)
        nums_radix = nums_scenario_5
        start_radix = time.process_time()
        radix.algorithm(nums_radix)
        end_radix = time.process_time()
        time_scenario_5["radix"].append(end_radix - start_radix)
        nums_bucket = nums_scenario_5
        start_bucket = time.process_time()
        bucket.algorithm(nums_bucket)
        end_bucket = time.process_time()
        time_scenario_5["bucket"].append(end_bucket - start_bucket)
        nums_tim = nums_scenario_5
        start_tim = time.process_time()
        tim.tim_sort(nums_tim)
        end_tim = time.process_time()
        time_scenario_5["tim"].append(end_tim - start_tim)
    elements=[]
    for i in range(1000, 10000, 1000):
        nums_scenario_6 = scenario_6(i)
        elements.append(len(nums_scenario_6))
        nums_quick = nums_scenario_6
        start_quick = time.process_time()
        quick.algorithm(nums_quick, 0, len(nums_quick) - 1)
        end_quick = time.process_time()
        time_scenario_6["quick"].append(end_quick - start_quick)
        nums_merge = nums_scenario_6
        start_merge = time.process_time()
        merge.algorithm(nums_merge)
        end_merge = time.process_time()
        time_scenario_6["merge"].append(end_merge - start_merge)
        nums_heap = nums_scenario_6
        start_heap = time.process_time()
        heap.algorithm(nums_heap)
        end_heap = time.process_time()
        time_scenario_6["heap"].append(end_heap - start_heap)
        nums_radix = nums_scenario_6
        start_radix = time.process_time()
        radix.algorithm(nums_radix)
        end_radix = time.process_time()
        time_scenario_6["radix"].append(end_radix - start_radix)
        nums_bucket = nums_scenario_6
        start_bucket = time.process_time()
        bucket.algorithm(nums_bucket)
        end_bucket = time.process_time()
        time_scenario_6["bucket"].append(end_bucket - start_bucket)
        nums_tim = nums_scenario_6
        start_tim = time.process_time()
        tim.tim_sort(nums_tim)
        end_tim = time.process_time()
        time_scenario_6["tim"].append(end_tim - start_tim)
    graph_scenario_1(time_scenario_1, elements)
    graph_scenario_2(time_scenario_2,elements)
    graph_scenario_3(time_scenario_3,elements)
    graph_scenario_4(time_scenario_4,elements)
    graph_scenario_5(time_scenario_5,elements)
    graph_scenario_6(time_scenario_6,elements)


senario_vs_sort_algorithm()
# Data structures to store results

times_quick = {"scenario_1": [], "scenario_2": [], "scenario_3": [], "scenario_4": [], "scenario_5": [],
               "scenario_6": []}
times_heap = {"scenario_1": [], "scenario_2": [], "scenario_3": [], "scenario_4": [], "scenario_5": [],
              "scenario_6": []}
times_merge = {"scenario_1": [], "scenario_2": [], "scenario_3": [], "scenario_4": [], "scenario_5": [],
               "scenario_6": []}
times_radix = {"scenario_1": [], "scenario_2": [], "scenario_3": [], "scenario_4": [], "scenario_5": [],
               "scenario_6": []}
times_bucket = {"scenario_1": [], "scenario_2": [], "scenario_3": [], "scenario_4": [], "scenario_5": [],
                "scenario_6": []}
times_tim_sort = {"scenario_1": [], "scenario_2": [], "scenario_3": [], "scenario_4": [], "scenario_5": [],
                  "scenario_6": []}

# please modify the for loop to start from 1000 to 20000 if you are  not running bucket sort else leave it as it is
print("senario_1")
# Scenario 1
elements = []
for i in range(1000, 5000, 1000):
    a = scenario_1(i)
    a_quick = a
    a_heap = a
    a_merge = a
    a_radix = a
    a_bucket = a
    a_tim = a
    start_tim = time.process_time()
    tim.tim_sort(a_tim)
    end_tim = time.process_time()
    times_tim_sort["scenario_1"].append(end_tim - start_tim)
    start = time.process_time()
    quick.algorithm(a_quick, 0, len(a) - 1)
    end = time.process_time()
    start_heap = time.process_time()
    heap.algorithm(a_heap)
    end_heap = time.process_time()
    start_merge = time.process_time()
    merge.algorithm(a_merge)
    end_merge = time.process_time()
    start_radix = time.process_time()
    radix.algorithm(a_radix)
    end_radix = time.process_time()
    start_bucket = time.process_time()
    bucket.algorithm(a_bucket)
    end_bucket = time.process_time()
    elements.append(len(a))
    times_radix["scenario_1"].append(end_radix - start_radix)
    times_bucket["scenario_1"].append(end_bucket - start_bucket)
    times_quick["scenario_1"].append(end - start)
    times_heap["scenario_1"].append(end_heap - start_heap)
    times_merge["scenario_1"].append(end_merge - start_merge)

# Data structures to store results
elements = []
# Scenario 2
print("senario_2")
for i in range(1000, 5000, 1000):
    a = scenario_2(i)
    a_quick = a
    a_heap = a
    a_merge = a
    a_radix = a
    a_bucket = a
    a_tim = a
    start_tim = time.process_time()
    tim.tim_sort(a_tim)
    end_tim = time.process_time()
    times_tim_sort["scenario_2"].append(end_tim - start_tim)
    start = time.process_time()
    quick.algorithm(a_quick, 0, len(a) - 1)
    end = time.process_time()
    start_heap = time.process_time()
    heap.algorithm(a_heap)
    end_heap = time.process_time()
    start_merge = time.process_time()
    merge.algorithm(a_merge)
    end_merge = time.process_time()
    start_radix = time.process_time()
    radix.algorithm(a_radix)
    end_radix = time.process_time()
    start_bucket = time.process_time()
    bucket.algorithm(a_bucket)
    end_bucket = time.process_time()
    elements.append(len(a))
    times_radix["scenario_2"].append(end_radix - start_radix)
    times_bucket["scenario_2"].append(end_bucket - start_bucket)
    times_quick["scenario_2"].append(end - start)
    times_heap["scenario_2"].append(end_heap - start_heap)
    times_merge["scenario_2"].append(end_merge - start_merge)
# Data structures to store results
elements = []
# Scenario 3
print("senario_3")
for i in range(1000, 5000, 1000):
    a = scenario_3(i)
    a_quick = a
    a_heap = a
    a_merge = a
    a_radix = a
    a_bucket = a
    a_tim = a
    start_tim = time.process_time()
    tim.tim_sort(a_tim)
    end_tim = time.process_time()
    times_tim_sort["scenario_3"].append(end_tim - start_tim)
    start = time.process_time()
    quick.algorithm(a_quick, 0, len(a) - 1)
    end = time.process_time()
    start_heap = time.process_time()
    heap.algorithm(a_heap.copy())
    end_heap = time.process_time()
    start_merge = time.process_time()
    merge.algorithm(a_merge)
    end_merge = time.process_time()
    start_radix = time.process_time()
    radix.algorithm(a_radix)
    end_radix = time.process_time()
    start_bucket = time.process_time()
    bucket.algorithm(a_bucket)
    end_bucket = time.process_time()
    elements.append(len(a))
    times_radix["scenario_3"].append(end_radix - start_radix)
    times_bucket["scenario_3"].append(end_bucket - start_bucket)
    times_quick["scenario_3"].append(end - start)
    times_heap["scenario_3"].append(end_heap - start_heap)
    times_merge["scenario_3"].append(end_merge - start_merge)
# Scenario 4
elements = []
print("senario_4")
for i in range(1000, 5000, 1000):
    a = scenario_4(i)
    a_quick = a
    a_heap = a
    a_merge = a
    a_radix = a
    a_bucket = a
    a_tim = a
    start_tim = time.process_time()
    tim.tim_sort(a_tim)
    end_tim = time.process_time()
    times_tim_sort["scenario_4"].append(end_tim - start_tim)
    start = time.process_time()
    quick.algorithm(a_quick, 0, len(a) - 1)
    end = time.process_time()
    start_heap = time.process_time()
    heap.algorithm(a_heap)
    end_heap = time.process_time()
    start_merge = time.process_time()
    merge.algorithm(a_merge)
    end_merge = time.process_time()
    start_radix = time.process_time()
    radix.algorithm(a_radix)
    end_radix = time.process_time()
    start_bucket = time.process_time()
    bucket.algorithm(a_bucket)
    end_bucket = time.process_time()
    elements.append(len(a))
    times_radix["scenario_4"].append(end_radix - start_radix)
    times_bucket["scenario_4"].append(end_bucket - start_bucket)
    times_quick["scenario_4"].append(end - start)
    times_heap["scenario_4"].append(end_heap - start_heap)
    times_merge["scenario_4"].append(end_merge - start_merge)
# Scenario 5
elements = []
for i in range(1000, 5000, 1000):
    a = scenario_5(i)
    a_quick = a
    a_heap = a
    a_merge = a
    a_radix = a
    a_bucket = a
    a_tim = a
    start_tim = time.process_time()
    tim.tim_sort(a_tim)
    end_tim = time.process_time()
    times_tim_sort["scenario_5"].append(end_tim - start_tim)
    start = time.process_time()
    quick.algorithm(a_quick, 0, len(a) - 1)
    end = time.process_time()
    start_heap = time.process_time()
    heap.algorithm(a_heap)
    end_heap = time.process_time()
    start_merge = time.process_time()
    merge.algorithm(a_merge)
    end_merge = time.process_time()
    start_radix = time.process_time()
    radix.algorithm(a_radix)
    end_radix = time.process_time()
    start_bucket = time.process_time()
    bucket.algorithm(a_bucket)
    end_bucket = time.process_time()
    elements.append(len(a))
    times_radix["scenario_5"].append(end_radix - start_radix)
    times_bucket["scenario_5"].append(end_bucket - start_bucket)
    times_quick["scenario_5"].append(end - start)
    times_heap["scenario_5"].append(end_heap - start_heap)
    times_merge["scenario_5"].append(end_merge - start_merge)
# Scenario 6
elements = []
print("senario_6")
for i in range(1000, 5000, 1000):
    a = scenario_6(i)
    a_quick = a
    a_heap = a
    a_merge = a
    a_radix = a
    a_bucket = a
    a_tim = a
    start_tim = time.process_time()
    tim.tim_sort(a_tim)
    end_tim = time.process_time()
    times_tim_sort["scenario_6"].append(end_tim - start_tim)
    start = time.process_time()
    quick.algorithm(a_quick, 0, len(a) - 1)
    end = time.process_time()
    start_heap = time.process_time()
    heap.algorithm(a_heap)
    end_heap = time.process_time()
    start_merge = time.process_time()
    merge.algorithm(a_merge)
    end_merge = time.process_time()
    start_radix = time.process_time()
    radix.algorithm(a_radix)
    end_radix = time.process_time()
    start_bucket = time.process_time()
    bucket.algorithm(a_bucket)
    end_bucket = time.process_time()
    elements.append(len(a))
    times_bucket["scenario_6"].append(end_bucket - start_bucket)
    times_radix["scenario_6"].append(end_radix - start_radix)
    times_quick["scenario_6"].append(end - start)
    times_heap["scenario_6"].append(end_heap - start_heap)
    times_merge["scenario_6"].append(end_merge - start_merge)

# Plotting
graph_quick_sort(times_quick,elements)
graph_heap_sort(times_heap,elements)
graph_merge_sort(times_merge,elements)
graph_radix_sort(times_radix,elements)
graph_bucket_sort(times_bucket,elements)
graph_tim_sort(times_tim_sort,elements)
