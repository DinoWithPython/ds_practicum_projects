n, count = list(map(int, input().split()))

lst = [int(x) for x in input().split()]
if count == 1:
	print(1)
if count == 2:
	print(lst[0] + lst[1])
if count > 2:
	need_index = lst.index(count)
    print(sum(lst[:need_index+1]))