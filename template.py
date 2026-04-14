import math
import re
import ast
import heapq
import bisect
import itertools
from collections import defaultdict, Counter, deque, OrderedDict

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Node:
    def __init__(self, val=0, next=None, random=None, left=None, right=None, children=None):
        self.val = val
        self.next = next
        self.random = random
        self.left = left
        self.right = right
        self.children = children or []

class ProblemBank:
    def __init__(self):
        self.routes = {}

    def register(self, *titles):
        def deco(fn):
            for t in titles:
                self.routes[self._norm(t)] = fn
            return fn
        return deco

    def _norm(self, title: str) -> str:
        title = title.lower().strip()
        title = re.sub(r'[^a-z0-9]+', ' ', title)
        return re.sub(r'\s+', ' ', title).strip()

    def solve(self, title: str, *args):
        fn = self.routes.get(self._norm(title))
        if not fn:
            return None
        return fn(*args)

BANK = ProblemBank()

def parse_arg(s: str):
    s = s.strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return s

def linked_list_to_list(head):
    out = []
    seen = set()
    while head and id(head) not in seen:
        seen.add(id(head))
        out.append(head.val)
        head = head.next
    return out

def build_linked_list(values):
    if not values:
        return None
    dummy = ListNode(0)
    cur = dummy
    for v in values:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def build_tree(values):
    if not values:
        return None
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[::-1]
    root = kids.pop()
    for node in nodes:
        if node:
            if kids:
                node.left = kids.pop()
            if kids:
                node.right = kids.pop()
    return root

def tree_to_level_list(root):
    if not root:
        return []
    out = []
    q = deque([root])
    while q:
        node = q.popleft()
        if node:
            out.append(node.val)
            q.append(node.left)
            q.append(node.right)
        else:
            out.append(None)
    while out and out[-1] is None:
        out.pop()
    return out

# =========================
# ARRAY / HASH / TWO POINTERS / SLIDING WINDOW
# =========================

@BANK.register("Two Sum")
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []

@BANK.register("Two Sum II - Input array is sorted")
def two_sum_ii(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        s = nums[l] + nums[r]
        if s == target:
            return [l + 1, r + 1]
        if s < target:
            l += 1
        else:
            r -= 1
    return []

@BANK.register("Contains Duplicate")
def contains_duplicate(nums):
    return len(nums) != len(set(nums))

@BANK.register("Contains Duplicate II")
def contains_duplicate_ii(nums, k):
    last = {}
    for i, x in enumerate(nums):
        if x in last and i - last[x] <= k:
            return True
        last[x] = i
    return False

@BANK.register("Contains Duplicate III")
def contains_duplicate_iii(nums, indexDiff, valueDiff):
    if indexDiff < 1 or valueDiff < 0:
        return False
    w = valueDiff + 1
    buckets = {}
    for i, x in enumerate(nums):
        b = x // w
        if x < 0:
            b -= 1
        if b in buckets:
            return True
        if b - 1 in buckets and abs(x - buckets[b - 1]) < w:
            return True
        if b + 1 in buckets and abs(x - buckets[b + 1]) < w:
            return True
        buckets[b] = x
        if i >= indexDiff:
            old = nums[i - indexDiff]
            ob = old // w
            if old < 0:
                ob -= 1
            del buckets[ob]
    return False

@BANK.register("Maximum Subarray")
def maximum_subarray(nums):
    cur = best = nums[0]
    for n in nums[1:]:
        cur = max(n, cur + n)
        best = max(best, cur)
    return best

@BANK.register("Container With Most Water")
def container_with_most_water(height):
    l, r = 0, len(height) - 1
    ans = 0
    while l < r:
        ans = max(ans, min(height[l], height[r]) * (r - l))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return ans

@BANK.register("Longest Substring Without Repeating Characters")
def longest_substring_without_repeating_characters(s):
    seen = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        best = max(best, right - left + 1)
    return best

@BANK.register("Minimum Size Subarray Sum")
def minimum_size_subarray_sum(target, nums):
    left = 0
    s = 0
    ans = math.inf
    for right, x in enumerate(nums):
        s += x
        while s >= target:
            ans = min(ans, right - left + 1)
            s -= nums[left]
            left += 1
    return 0 if ans == math.inf else ans

@BANK.register("Subarray Sum Equals K")
def subarray_sum_equals_k(nums, k):
    cnt = Counter({0: 1})
    s = ans = 0
    for x in nums:
        s += x
        ans += cnt[s - k]
        cnt[s] += 1
    return ans

@BANK.register("Product of Array Except Self")
def product_of_array_except_self(nums):
    n = len(nums)
    res = [1] * n
    pre = 1
    for i in range(n):
        res[i] = pre
        pre *= nums[i]
    suf = 1
    for i in range(n - 1, -1, -1):
        res[i] *= suf
        suf *= nums[i]
    return res

@BANK.register("Top K Frequent Elements")
def top_k_frequent_elements(nums, k):
    cnt = Counter(nums)
    return [x for x, _ in heapq.nlargest(k, cnt.items(), key=lambda p: p[1])]

@BANK.register("Group Anagrams")
def group_anagrams(strs):
    mp = defaultdict(list)
    for s in strs:
        mp[tuple(sorted(s))].append(s)
    return list(mp.values())

@BANK.register("Longest Consecutive Sequence")
def longest_consecutive_sequence(nums):
    st = set(nums)
    ans = 0
    for x in st:
        if x - 1 not in st:
            y = x
            while y in st:
                y += 1
            ans = max(ans, y - x)
    return ans

@BANK.register("Move Zeroes")
def move_zeroes(nums):
    k = 0
    for x in nums:
        if x != 0:
            nums[k] = x
            k += 1
    for i in range(k, len(nums)):
        nums[i] = 0
    return nums

@BANK.register("Best Time to Buy and Sell Stock")
def best_time_to_buy_and_sell_stock(prices):
    mn = math.inf
    ans = 0
    for p in prices:
        mn = min(mn, p)
        ans = max(ans, p - mn)
    return ans

@BANK.register("Best Time to Buy and Sell Stock II")
def best_time_to_buy_and_sell_stock_ii(prices):
    return sum(max(0, prices[i + 1] - prices[i]) for i in range(len(prices) - 1))

@BANK.register("Best Time to Buy and Sell Stock III")
def best_time_to_buy_and_sell_stock_iii(prices):
    buy1 = buy2 = math.inf
    profit1 = profit2 = 0
    for p in prices:
        buy1 = min(buy1, p)
        profit1 = max(profit1, p - buy1)
        buy2 = min(buy2, p - profit1)
        profit2 = max(profit2, p - buy2)
    return profit2

@BANK.register("Merge Intervals")
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    res = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= res[-1][1]:
            res[-1][1] = max(res[-1][1], e)
        else:
            res.append([s, e])
    return res

@BANK.register("Insert Interval")
def insert_interval(intervals, newInterval):
    res = []
    i = 0
    while i < len(intervals) and intervals[i][1] < newInterval[0]:
        res.append(intervals[i])
        i += 1
    while i < len(intervals) and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    res.append(newInterval)
    while i < len(intervals):
        res.append(intervals[i])
        i += 1
    return res

@BANK.register("Jump Game")
def jump_game(nums):
    far = 0
    for i, x in enumerate(nums):
        if i > far:
            return False
        far = max(far, i + x)
    return True

@BANK.register("Jump Game II")
def jump_game_ii(nums):
    end = far = ans = 0
    for i in range(len(nums) - 1):
        far = max(far, i + nums[i])
        if i == end:
            ans += 1
            end = far
    return ans

@BANK.register("Rotate Image")
def rotate_image(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
    return matrix

@BANK.register("Spiral Matrix")
def spiral_matrix(matrix):
    res = []
    while matrix:
        res += matrix.pop(0)
        if matrix and matrix[0]:
            for row in matrix:
                res.append(row.pop())
        if matrix:
            res += matrix.pop()[::-1]
        if matrix and matrix[0]:
            for row in matrix[::-1]:
                res.append(row.pop(0))
    return res

@BANK.register("Spiral Matrix II")
def spiral_matrix_ii(n):
    mat = [[0] * n for _ in range(n)]
    left = top = 0
    right = bottom = n - 1
    cur = 1
    while left <= right and top <= bottom:
        for c in range(left, right + 1):
            mat[top][c] = cur
            cur += 1
        top += 1
        for r in range(top, bottom + 1):
            mat[r][right] = cur
            cur += 1
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                mat[bottom][c] = cur
                cur += 1
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                mat[r][left] = cur
                cur += 1
            left += 1
    return mat

@BANK.register("Search a 2D Matrix")
def search_a_2d_matrix(matrix, target):
    flat = [x for row in matrix for x in row]
    i = bisect.bisect_left(flat, target)
    return i < len(flat) and flat[i] == target

@BANK.register("Search a 2D Matrix II")
def search_a_2d_matrix_ii(matrix, target):
    if not matrix:
        return False
    r, c = 0, len(matrix[0]) - 1
    while r < len(matrix) and c >= 0:
        if matrix[r][c] == target:
            return True
        if matrix[r][c] > target:
            c -= 1
        else:
            r += 1
    return False

@BANK.register("Set Matrix Zeroes")
def set_matrix_zeroes(matrix):
    rows = set()
    cols = set()
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            if matrix[r][c] == 0:
                rows.add(r)
                cols.add(c)
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            if r in rows or c in cols:
                matrix[r][c] = 0
    return matrix

@BANK.register("Sort Colors")
def sort_colors(nums):
    cnt = Counter(nums)
    i = 0
    for x in [0, 1, 2]:
        for _ in range(cnt[x]):
            nums[i] = x
            i += 1
    return nums

@BANK.register("Find First and Last Position of Element in Sorted Array")
def find_first_and_last_position_of_element_in_sorted_array(nums, target):
    l = bisect.bisect_left(nums, target)
    r = bisect.bisect_right(nums, target) - 1
    if l <= r:
        return [l, r]
    return [-1, -1]

@BANK.register("Search in Rotated Sorted Array")
def search_in_rotated_sorted_array(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            return m
        if nums[l] <= nums[m]:
            if nums[l] <= target < nums[m]:
                r = m - 1
            else:
                l = m + 1
        else:
            if nums[m] < target <= nums[r]:
                l = m + 1
            else:
                r = m - 1
    return -1

@BANK.register("Search in Rotated Sorted Array II")
def search_in_rotated_sorted_array_ii(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            return True
        if nums[l] == nums[m] == nums[r]:
            l += 1
            r -= 1
        elif nums[l] <= nums[m]:
            if nums[l] <= target < nums[m]:
                r = m - 1
            else:
                l = m + 1
        else:
            if nums[m] < target <= nums[r]:
                l = m + 1
            else:
                r = m - 1
    return False

@BANK.register("Search Insert Position")
def search_insert_position(nums, target):
    return bisect.bisect_left(nums, target)

@BANK.register("Remove Duplicates from Sorted Array")
def remove_duplicates_from_sorted_array(nums):
    if not nums:
        return 0
    k = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[k - 1]:
            nums[k] = nums[i]
            k += 1
    return k

@BANK.register("Remove Duplicates from Sorted Array II")
def remove_duplicates_from_sorted_array_ii(nums):
    k = 0
    for x in nums:
        if k < 2 or x != nums[k - 2]:
            nums[k] = x
            k += 1
    return k

@BANK.register("Remove Element")
def remove_element(nums, val):
    k = 0
    for x in nums:
        if x != val:
            nums[k] = x
            k += 1
    return k

@BANK.register("Plus One")
def plus_one(digits):
    i = len(digits) - 1
    while i >= 0 and digits[i] == 9:
        digits[i] = 0
        i -= 1
    if i < 0:
        return [1] + digits
    digits[i] += 1
    return digits

@BANK.register("Add Binary")
def add_binary(a, b):
    return bin(int(a, 2) + int(b, 2))[2:]

@BANK.register("Length of Last Word")
def length_of_last_word(s):
    s = s.strip()
    return 0 if not s else len(s.split()[-1])

@BANK.register("Valid Parentheses")
def valid_parentheses(s):
    stack = []
    mp = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in mp:
            if not stack or stack.pop() != mp[ch]:
                return False
        else:
            stack.append(ch)
    return not stack

@BANK.register("Simplify Path")
def simplify_path(path):
    stack = []
    for p in path.split('/'):
        if p in ('', '.'):
            continue
        if p == '..':
            if stack:
                stack.pop()
        else:
            stack.append(p)
    return '/' + '/'.join(stack)

# =========================
# LINKED LIST
# =========================

@BANK.register("Merge Two Sorted Lists")
def merge_two_sorted_lists(l1, l2):
    dummy = cur = ListNode(0)
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next

@BANK.register("Remove Nth Node From End of List")
def remove_nth_node_from_end_of_list(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n):
        fast = fast.next
    while fast and fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next

@BANK.register("Swap Nodes in Pairs")
def swap_nodes_in_pairs(head):
    dummy = ListNode(0, head)
    prev = dummy
    while prev.next and prev.next.next:
        a = prev.next
        b = a.next
        prev.next, b.next, a.next = b, a, b.next
        prev = a
    return dummy.next

@BANK.register("Reverse Linked List")
def reverse_linked_list(head):
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev

@BANK.register("Palindrome Linked List")
def palindrome_linked_list(head):
    vals = linked_list_to_list(head)
    return vals == vals[::-1]

@BANK.register("Delete Node in a Linked List")
def delete_node_in_a_linked_list(node):
    node.val = node.next.val
    node.next = node.next.next

@BANK.register("Rotate List")
def rotate_list(head, k):
    if not head or not head.next:
        return head
    n = 1
    tail = head
    while tail.next:
        tail = tail.next
        n += 1
    tail.next = head
    k %= n
    steps = n - k
    cur = head
    for _ in range(steps - 1):
        cur = cur.next
    new_head = cur.next
    cur.next = None
    return new_head

@BANK.register("Remove Linked List Elements")
def remove_linked_list_elements(head, val):
    dummy = ListNode(0, head)
    cur = dummy
    while cur.next:
        if cur.next.val == val:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return dummy.next

@BANK.register("Intersection of Two Linked Lists")
def intersection_of_two_linked_lists(headA, headB):
    a, b = headA, headB
    while a is not b:
        a = a.next if a else headB
        b = b.next if b else headA
    return a

@BANK.register("Reverse Nodes in k-Group")
def reverse_nodes_in_k_group(head, k):
    def get_kth(cur, k):
        while cur and k > 0:
            cur = cur.next
            k -= 1
        return cur

    dummy = ListNode(0, head)
    group_prev = dummy
    while True:
        kth = get_kth(group_prev, k)
        if not kth:
            break
        group_next = kth.next
        prev, cur = kth.next, group_prev.next
        while cur != group_next:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        tmp = group_prev.next
        group_prev.next = kth
        group_prev = tmp
    return dummy.next

@BANK.register("Remove Duplicates from Sorted List")
def remove_duplicates_from_sorted_list(head):
    cur = head
    while cur and cur.next:
        if cur.val == cur.next.val:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return head

@BANK.register("Partition List")
def partition_list(head, x):
    small = s_tail = ListNode(0)
    large = l_tail = ListNode(0)
    while head:
        nxt = head.next
        head.next = None
        if head.val < x:
            s_tail.next = head
            s_tail = s_tail.next
        else:
            l_tail.next = head
            l_tail = l_tail.next
        head = nxt
    s_tail.next = large.next
    return small.next

@BANK.register("Merge Sorted Array")
def merge_sorted_array(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    return nums1

@BANK.register("Odd Even Linked List")
def odd_even_linked_list(head):
    if not head or not head.next:
        return head
    odd = head
    even = even_head = head.next
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    odd.next = even_head
    return head

# =========================
# TREES / BST
# =========================

@BANK.register("Binary Tree Inorder Traversal")
def binary_tree_inorder_traversal(root):
    res = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        res.append(node.val)
        dfs(node.right)
    dfs(root)
    return res

@BANK.register("Binary Tree Preorder Traversal")
def binary_tree_preorder_traversal(root):
    res = []
    def dfs(node):
        if not node:
            return
        res.append(node.val)
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return res

@BANK.register("Binary Tree Postorder Traversal")
def binary_tree_postorder_traversal(root):
    res = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        dfs(node.right)
        res.append(node.val)
    dfs(root)
    return res

@BANK.register("Same Tree")
def same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return same_tree(p.left, q.left) and same_tree(p.right, q.right)

@BANK.register("Symmetric Tree")
def symmetric_tree(root):
    def mirror(a, b):
        if not a and not b:
            return True
        if not a or not b or a.val != b.val:
            return False
        return mirror(a.left, b.right) and mirror(a.right, b.left)
    return mirror(root, root)

@BANK.register("Maximum Depth of Binary Tree")
def maximum_depth_of_binary_tree(root):
    if not root:
        return 0
    return 1 + max(maximum_depth_of_binary_tree(root.left), maximum_depth_of_binary_tree(root.right))

@BANK.register("Minimum Depth of Binary Tree")
def minimum_depth_of_binary_tree(root):
    if not root:
        return 0
    q = deque([(root, 1)])
    while q:
        node, d = q.popleft()
        if not node.left and not node.right:
            return d
        if node.left:
            q.append((node.left, d + 1))
        if node.right:
            q.append((node.right, d + 1))

@BANK.register("Balanced Binary Tree")
def balanced_binary_tree(root):
    def height(node):
        if not node:
            return 0
        lh = height(node.left)
        if lh == -1:
            return -1
        rh = height(node.right)
        if rh == -1:
            return -1
        if abs(lh - rh) > 1:
            return -1
        return 1 + max(lh, rh)
    return height(root) != -1

@BANK.register("Path Sum")
def path_sum(root, targetSum):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == targetSum
    rem = targetSum - root.val
    return path_sum(root.left, rem) or path_sum(root.right, rem)

@BANK.register("Binary Tree Level Order Traversal")
def binary_tree_level_order_traversal(root):
    if not root:
        return []
    res = []
    q = deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res

@BANK.register("Binary Tree Level Order Traversal II")
def binary_tree_level_order_traversal_ii(root):
    return binary_tree_level_order_traversal(root)[::-1]

@BANK.register("Binary Tree Zigzag Level Order Traversal")
def binary_tree_zigzag_level_order_traversal(root):
    levels = binary_tree_level_order_traversal(root)
    for i in range(1, len(levels), 2):
        levels[i].reverse()
    return levels

@BANK.register("Invert Binary Tree")
def invert_binary_tree(root):
    if not root:
        return None
    root.left, root.right = invert_binary_tree(root.right), invert_binary_tree(root.left)
    return root

@BANK.register("Validate Binary Search Tree")
def validate_binary_search_tree(root):
    def dfs(node, lo, hi):
        if not node:
            return True
        if not (lo < node.val < hi):
            return False
        return dfs(node.left, lo, node.val) and dfs(node.right, node.val, hi)
    return dfs(root, -math.inf, math.inf)

@BANK.register("Kth Smallest Element in a BST")
def kth_smallest_element_in_a_bst(root, k):
    stack = []
    while True:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if k == 0:
            return root.val
        root = root.right

@BANK.register("Minimum Absolute Difference in BST")
def minimum_absolute_difference_in_bst(root):
    prev = None
    ans = math.inf
    def dfs(node):
        nonlocal prev, ans
        if not node:
            return
        dfs(node.left)
        if prev is not None:
            ans = min(ans, node.val - prev)
        prev = node.val
        dfs(node.right)
    dfs(root)
    return ans

@BANK.register("Binary Tree Right Side View")
def binary_tree_right_side_view(root):
    if not root:
        return []
    res = []
    q = deque([root])
    while q:
        for i in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            if i == 0:
                last = node.val
        res.append(last)
    return res

@BANK.register("Find Mode in Binary Search Tree")
def find_mode_in_binary_search_tree(root):
    cnt = Counter()
    def dfs(node):
        if not node:
            return
        cnt[node.val] += 1
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    mx = max(cnt.values()) if cnt else 0
    return [k for k, v in cnt.items() if v == mx]

@BANK.register("Binary Tree Paths")
def binary_tree_paths(root):
    res = []
    def dfs(node, path):
        if not node:
            return
        if not node.left and not node.right:
            res.append(path + [str(node.val)])
            return
        dfs(node.left, path + [str(node.val)])
        dfs(node.right, path + [str(node.val)])
    dfs(root, [])
    return ['->'.join(p) for p in res]

@BANK.register("Diameter of Binary Tree")
def diameter_of_binary_tree(root):
    ans = 0
    def height(node):
        nonlocal ans
        if not node:
            return 0
        l = height(node.left)
        r = height(node.right)
        ans = max(ans, l + r)
        return 1 + max(l, r)
    height(root)
    return ans

@BANK.register("Binary Tree Maximum Path Sum")
def binary_tree_maximum_path_sum(root):
    ans = -math.inf
    def dfs(node):
        nonlocal ans
        if not node:
            return 0
        l = max(0, dfs(node.left))
        r = max(0, dfs(node.right))
        ans = max(ans, node.val + l + r)
        return node.val + max(l, r)
    dfs(root)
    return ans

@BANK.register("Convert BST to Greater Tree")
def convert_bst_to_greater_tree(root):
    total = 0
    def dfs(node):
        nonlocal total
        if not node:
            return
        dfs(node.right)
        total += node.val
        node.val = total
        dfs(node.left)
    dfs(root)
    return root

@BANK.register("Binary Search Tree Iterator")
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        self._push_left(root)
    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left
    def next(self):
        node = self.stack.pop()
        self._push_left(node.right)
        return node.val
    def hasNext(self):
        return bool(self.stack)

@BANK.register("Lowest Common Ancestor of Deepest Leaves")
def lowest_common_ancestor_of_deepest_leaves(root):
    def dfs(node):
        if not node:
            return (0, None)
        ld, lca = dfs(node.left)
        rd, rca = dfs(node.right)
        if ld > rd:
            return (ld + 1, lca)
        if rd > ld:
            return (rd + 1, rca)
        return (ld + 1, node)
    return dfs(root)[1]

# =========================
# GRAPH / DFS / BFS
# =========================

@BANK.register("Number of Islands")
def number_of_islands(grid):
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    def dfs(r, c):
        if r < 0 or c < 0 or r >= m or c >= n or grid[r][c] == '0':
            return
        grid[r][c] = '0'
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    ans = 0
    for r in range(m):
        for c in range(n):
            if grid[r][c] == '1':
                ans += 1
                dfs(r, c)
    return ans

@BANK.register("Max Area of Island")
def max_area_of_island(grid):
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    def dfs(r, c):
        if r < 0 or c < 0 or r >= m or c >= n or grid[r][c] == 0:
            return 0
        grid[r][c] = 0
        return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)
    ans = 0
    for r in range(m):
        for c in range(n):
            if grid[r][c] == 1:
                ans = max(ans, dfs(r, c))
    return ans

@BANK.register("Keys and Rooms")
def keys_and_rooms(rooms):
    seen = {0}
    q = deque([0])
    while q:
        r = q.popleft()
        for k in rooms[r]:
            if k not in seen:
                seen.add(k)
                q.append(k)
    return len(seen) == len(rooms)

@BANK.register("Is Graph Bipartite?")
def is_graph_bipartite(graph):
    color = {}
    for i in range(len(graph)):
        if i in color:
            continue
        q = deque([i])
        color[i] = 0
        while q:
            u = q.popleft()
            for v in graph[u]:
                if v in color:
                    if color[v] == color[u]:
                        return False
                else:
                    color[v] = 1 - color[u]
                    q.append(v)
    return True

@BANK.register("Rotting Oranges")
def rotting_oranges(grid):
    m, n = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for r in range(m):
        for c in range(n):
            if grid[r][c] == 2:
                q.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1
    ans = 0
    while q:
        r, c, d = q.popleft()
        ans = max(ans, d)
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                q.append((nr, nc, d + 1))
    return ans if fresh == 0 else -1

@BANK.register("Shortest Path in Binary Matrix")
def shortest_path_in_binary_matrix(grid):
    n = len(grid)
    if grid[0][0] or grid[n-1][n-1]:
        return -1
    q = deque([(0,0,1)])
    grid[0][0] = 1
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while q:
        r, c, d = q.popleft()
        if r == n - 1 and c == n - 1:
            return d
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                grid[nr][nc] = 1
                q.append((nr, nc, d + 1))
    return -1

@BANK.register("Word Break")
def word_break(s, wordDict):
    wd = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in wd:
                dp[i] = True
                break
    return dp[-1]

@BANK.register("Maximal Square")
def maximal_square(matrix):
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ans = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i - 1][j - 1] == '1' or matrix[i - 1][j - 1] == 1:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                ans = max(ans, dp[i][j])
    return ans * ans

@BANK.register("Minimum Path Sum")
def minimum_path_sum(grid):
    m, n = len(grid), len(grid[0])
    for i in range(1, m):
        grid[i][0] += grid[i - 1][0]
    for j in range(1, n):
        grid[0][j] += grid[0][j - 1]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
    return grid[-1][-1]

@BANK.register("Unique Paths II")
def unique_paths_ii(obstacleGrid):
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [0] * n
    dp[0] = 1 if obstacleGrid[0][0] == 0 else 0
    for r in range(m):
        for c in range(n):
            if obstacleGrid[r][c] == 1:
                dp[c] = 0
            elif c > 0:
                dp[c] += dp[c - 1]
    return dp[-1]

@BANK.register("Triangle")
def triangle(tri):
    dp = tri[-1][:]
    for r in range(len(tri) - 2, -1, -1):
        for c in range(len(tri[r])):
            dp[c] = tri[r][c] + min(dp[c], dp[c + 1])
    return dp[0]

@BANK.register("Pascal's Triangle")
def pascals_triangle(numRows):
    res = []
    for _ in range(numRows):
        row = [1]
        if res:
            prev = res[-1]
            row += [prev[i] + prev[i + 1] for i in range(len(prev) - 1)]
            row.append(1)
        res.append(row)
    return res

@BANK.register("Pascal's Triangle II")
def pascals_triangle_ii(rowIndex):
    row = [1]
    for _ in range(rowIndex):
        row = [1] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1]
    return row

# =========================
# STRING / MATH
# =========================

@BANK.register("Reverse Integer")
def reverse_integer(x):
    sign = -1 if x < 0 else 1
    x = abs(x)
    y = int(str(x)[::-1]) * sign
    return y if -2**31 <= y <= 2**31 - 1 else 0

@BANK.register("Palindrome Number")
def palindrome_number(x):
    if x < 0:
        return False
    original, rev = x, 0
    while x:
        rev = rev * 10 + x % 10
        x //= 10
    return original == rev

@BANK.register("Roman to Integer")
def roman_to_integer(s):
    mp = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    for i, ch in enumerate(s):
        if i + 1 < len(s) and mp[ch] < mp[s[i + 1]]:
            total -= mp[ch]
        else:
            total += mp[ch]
    return total

@BANK.register("Integer to Roman")
def integer_to_roman(num):
    vals = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    out = []
    for v, s in zip(vals, syms):
        while num >= v:
            num -= v
            out.append(s)
    return ''.join(out)

@BANK.register("Longest Common Prefix")
def longest_common_prefix(strs):
    if not strs:
        return ""
    pref = strs[0]
    for s in strs[1:]:
        while not s.startswith(pref):
            pref = pref[:-1]
            if not pref:
                return ""
    return pref

@BANK.register("Valid Anagram")
def valid_anagram(s, t):
    return Counter(s) == Counter(t)

@BANK.register("Happy Number")
def happy_number(n):
    seen = set()
    def nxt(x):
        s = 0
        while x:
            x, d = divmod(x, 10)
            s += d * d
        return s
    while n != 1 and n not in seen:
        seen.add(n)
        n = nxt(n)
    return n == 1

@BANK.register("Single Number")
def single_number(nums):
    x = 0
    for n in nums:
        x ^= n
    return x

@BANK.register("Single Number II")
def single_number_ii(nums):
    ones = twos = 0
    for n in nums:
        ones = (ones ^ n) & ~twos
        twos = (twos ^ n) & ~ones
    return ones

@BANK.register("Count Primes")
def count_primes(n):
    if n < 3:
        return 0
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n, i):
                sieve[j] = False
    return sum(sieve)

@BANK.register("Factorial Trailing Zeroes")
def factorial_trailing_zeroes(n):
    ans = 0
    while n:
        n //= 5
        ans += n
    return ans

@BANK.register("Number of 1 Bits")
def number_of_1_bits(n):
    return bin(n & 0xffffffff).count('1')

@BANK.register("Power of Three")
def power_of_three(n):
    while n > 1 and n % 3 == 0:
        n //= 3
    return n == 1

@BANK.register("Power of Four")
def power_of_four(n):
    return n > 0 and (n & (n - 1)) == 0 and n % 3 == 1

@BANK.register("Add Digits")
def add_digits(num):
    return 0 if num == 0 else 1 + (num - 1) % 9

@BANK.register("Excel Sheet Column Title")
def excel_sheet_column_title(n):
    out = []
    while n:
        n -= 1
        n, r = divmod(n, 26)
        out.append(chr(65 + r))
    return ''.join(reversed(out))

@BANK.register("Excel Sheet Column Number")
def excel_sheet_column_number(s):
    ans = 0
    for ch in s:
        ans = ans * 26 + (ord(ch) - 64)
    return ans

# =========================
# STRUCTURES
# =========================

@BANK.register("LRU Cache")
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.od = OrderedDict()
    def get(self, key):
        if key not in self.od:
            return -1
        self.od.move_to_end(key)
        return self.od[key]
    def put(self, key, value):
        if key in self.od:
            self.od.move_to_end(key)
        self.od[key] = value
        if len(self.od) > self.cap:
            self.od.popitem(last=False)

@BANK.register("Min Stack")
class MinStack:
    def __init__(self):
        self.s = []
        self.m = []
    def push(self, val):
        self.s.append(val)
        self.m.append(val if not self.m else min(val, self.m[-1]))
    def pop(self):
        self.s.pop()
        self.m.pop()
    def top(self):
        return self.s[-1]
    def getMin(self):
        return self.m[-1]

@BANK.register("Implement Stack using Queues")
class MyStack:
    def __init__(self):
        self.q = deque()
    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
    def pop(self):
        return self.q.popleft()
    def top(self):
        return self.q[0]
    def empty(self):
        return not self.q

@BANK.register("Implement Queue using Stacks")
class MyQueue:
    def __init__(self):
        self.a = []
        self.b = []
    def push(self, x):
        self.a.append(x)
    def pop(self):
        self.peek()
        return self.b.pop()
    def peek(self):
        if not self.b:
            while self.a:
                self.b.append(self.a.pop())
        return self.b[-1]
    def empty(self):
        return not self.a and not self.b

@BANK.register("Implement Trie (Prefix Tree)")
class Trie:
    def __init__(self):
        self.children = {}
        self.end = False
    def insert(self, word):
        node = self
        for ch in word:
            node = node.children.setdefault(ch, Trie())
        node.end = True
    def search(self, word):
        node = self
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.end
    def startsWith(self, prefix):
        node = self
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

@BANK.register("Add and Search Word - Data structure design")
class WordDictionary:
    def __init__(self):
        self.root = {}
        self.end = '#'
    def addWord(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node[self.end] = True
    def search(self, word):
        def dfs(i, node):
            if i == len(word):
                return self.end in node
            ch = word[i]
            if ch == '.':
                return any(dfs(i + 1, nxt) for k, nxt in node.items() if k != self.end)
            return ch in node and dfs(i + 1, node[ch])
        return dfs(0, self.root)

@BANK.register("Design Parking System")
class ParkingSystem:
    def __init__(self, big, medium, small):
        self.spots = {1: big, 2: medium, 3: small}
    def addCar(self, carType):
        if self.spots[carType] <= 0:
            return False
        self.spots[carType] -= 1
        return True

@BANK.register("Design Circular Queue")
class MyCircularQueue:
    def __init__(self, k):
        self.q = deque()
        self.k = k
    def enQueue(self, value):
        if len(self.q) >= self.k:
            return False
        self.q.append(value)
        return True
    def deQueue(self):
        if not self.q:
            return False
        self.q.popleft()
        return True
    def Front(self):
        return self.q[0] if self.q else -1
    def Rear(self):
        return self.q[-1] if self.q else -1
    def isEmpty(self):
        return not self.q
    def isFull(self):
        return len(self.q) == self.k

# =========================
# RUN
# =========================

def run_problem(title: str, args: list):
    res = BANK.solve(title, *args)
    if res is None:
        print("Problem not implemented yet:", title)
        return
    if isinstance(res, ListNode):
        print(linked_list_to_list(res))
    else:
        print(res)

if __name__ == "__main__":
    print("Template loaded")
    while True:
        title = input("\nProblem title: ").strip()
        if not title:
            break
        args = []
        i = 1
        while True:
            raw = input(f"Arg {i}: ").strip()
            if not raw:
                break
            args.append(parse_arg(raw))
            i += 1
        run_problem(title, args)
