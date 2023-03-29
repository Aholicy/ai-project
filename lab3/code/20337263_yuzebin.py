import random

sd = int(input("seed="))
random.seed(sd)
lst = [random.randint(1, 100) for _ in range(20)]
lst.sort()
first_half = lst[:10]
second_half = lst[10:]
first_half.sort()
second_half.sort(reverse=True)
sorted_lst = first_half + second_half
print(sorted_lst)

random.seed(sd)
lst2 = [random.randint(1, 100) for _ in range(20)]
lst2.sort()
first_half2 = lst2[:10]
second_half2 = lst2[10:]
first_half2.sort()
second_half2.sort(reverse=True)
sorted_lst2 = first_half2 + second_half2
print(sorted_lst2)