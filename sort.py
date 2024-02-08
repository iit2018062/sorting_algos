import random
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import sys
# when dealing with bucket short the array size greater than 2000 seems to take a lot of time on my system and freeze
# the computer
sys.setrecursionlimit(10 ** 6)
MIN_MERGE = 32


# quick sort algorithm below
class QuickSort:
    def partition(self, nums, start, end):
        i = start - 1
        left = nums[end]
        j = start
        while j <= end:
            if nums[j] <= left:
                i += 1
                if i < j:
                    nums[i], nums[j] = nums[j], nums[i]
            j += 1
        return i

    def algorithm(self, nums, start, end):
        if start < end:
            pivot = self.partition(nums, start, end)
            self.algorithm(nums, start, pivot - 1)
            self.algorithm(nums, pivot + 1, end)


# heap sort algorithm below
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


# merge sort algorithm below
class MergeSort():

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


# radix sort below
class RadixSort():

    def algorithm(self, array):

        def radix_sort(self, exp):
            output = [0] * len(array)
            frequency = [0] * (10)
            for i in range(0, len(array)):
                idx = (array[i] // exp)
                frequency[int((idx) % 10)] += 1
            for i in range(1, 10):
                frequency[i] += frequency[i - 1]
            i = len(array) - 1
            while i >= 0:
                idx = (array[i] / exp)
                output[frequency[int((idx) % 10)] - 1] = array[i]
                frequency[int((idx) % 10)] -= 1
                i -= 1
            i = 0
            for i in range(len(array)):
                array[i] = output[i]

        maximum = max(array)
        exp = 1
        while maximum // exp > 0:
            radix_sort(self, exp)
            exp *= 10


# bucket sort below
class BucketSort():
    def insertion_sort(self, b):
        for i in range(1, len(b)):
            up = b[i]
            j = i - 1
            while j >= 0 and b[j] > up:
                b[j + 1] = b[j]
                j -= 1
            b[j + 1] = up
        return b

    def algorithm(self, array):
        bucket_size = 100
        min = array[0]
        max = array[0]
        for i in range(1, len(array)):
            if array[i] < min:
                min = array[i]
            elif array[i] > max:
                max = array[i]
        bucket_count = ((max - min) // bucket_size) + 1
        buckets = []
        for i in range(0, bucket_count):
            buckets.append([])
        for i in range(0, len(array)):
            buckets[(array[i] - min) // bucket_size].append(array[i])
        k = 0
        for i in range(0, len(buckets)):
            self.insertion_sort(buckets[i])
            for j in range(0, len(buckets[i])):
                array[k] = buckets[i][j]
                k += 1


# tim sort below
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
    return randint(0, 1000 * size, size=size)


# [0,k], k<1000
def scenario_2(size):
    return randint(0, 1000, size=size)


# [0,n^3]
def scenario_3(size):
    return randint(0, size ** 3, size=size)


# [0,log(n)]
def scenario_4(size):
    return randint(0, int(np.log2(size)), size=size)


# [0,n](*1000)
def scenario_5(size):
    return [np.random.randint(0, size) * 1000 for _ in range(size)]


# [0,n](swapped)
def scenario_6(size):
    number_swaps = int((np.log2(size) / 2) // 1)
    sequence = list(range(size + 1))
    for i in range(number_swaps):
        index1, index2 = random.sample(range(size + 1), 2)
        sequence[index1], sequence[index2] = sequence[index2], sequence[index1]
    return sequence


def graph_quick_sort(times_quick):
    for scenario, times in times_quick.items():
        plt.plot(elements, times, label=f'QuickSort - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()


def graph_heap_sort(times_heap):
    for scenario, times in times_heap.items():
        plt.plot(elements, times, label=f'heapsort - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()


def graph_merge_sort(times_merge):
    for scenario, times in times_merge.items():
        plt.plot(elements, times, label=f'mergeSort - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()


def graph_radix_sort(times_radix):
    for scenario, times in times_radix.items():
        plt.plot(elements, times, label=f'radixSort - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()


def graph_bucket_sort(times_bucket):
    for scenario, times in times_bucket.items():
        plt.plot(elements, times, label=f'bucketSort - {scenario}')
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.grid()
    plt.legend()
    plt.show()


def graph_tim_sort(times_tim_sort):
    for scenario, times in times_tim_sort.items():
        plt.plot(elements, times, label=f'timSort - {scenario}')
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
bucket = BucketSort()
tim = TimSort()
# Data structures to store results
elements = []
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
for i in range(100, 1000, 100):
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
for i in range(100, 1000, 100):
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
for i in range(100, 1000, 100):
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
for i in range(100, 1000, 100):
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
print("senario_5")
for i in range(100, 1000, 100):
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
for i in range(100, 1000, 100):
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
graph_quick_sort(times_quick)
graph_heap_sort(times_heap)
graph_merge_sort(times_merge)
graph_radix_sort(times_radix)
graph_bucket_sort(times_bucket)
graph_tim_sort(times_tim_sort)
