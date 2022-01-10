# 1. Numpy ndarray 개요

***

### ndarray 생성 np.array()


```python
import numpy as np
```


```python
list1 = [1, 2, 3]
print("list1 : {}".format(list1))
print("list1 type : {}".format(type(list1)))

array1 = np.array(list1)
print("array1 : {}".format(array1))
print("array1 type : {}".format(type(array1)))
```

    list1 : [1, 2, 3]
    list1 type : <class 'list'>
    array1 : [1 2 3]
    array1 type : <class 'numpy.ndarray'>
    

___

### ndarray의 형태(shape)와 차원


```python
list1 = [1, 2, 3]
list2 = [2, 3, 4]

array1 = np.array(list1)
array2 = np.array([list1, list2])
array3 = np.array([list1])

print("array1 : {}".format(array1))
print("array1 type : {}".format(type(array1)))
print("array1 shape : {}".format(array1.shape))
      
print("array2 : {}".format(array2))
print("array2 type : {}".format(type(array2)))
print("array3 shape : {}".format(array1.shape))
      
print("array3 : {}".format(array3))
print("array3 type : {}".format(type(array3)))
print("array3 shape : {}".format(array1.shape))


```

    array1 : [1 2 3]
    array1 type : <class 'numpy.ndarray'>
    array1 shape : (3,)
    array2 : [[1 2 3]
     [2 3 4]]
    array2 type : <class 'numpy.ndarray'>
    array3 shape : (3,)
    array3 : [[1 2 3]]
    array3 type : <class 'numpy.ndarray'>
    array3 shape : (3,)
    


```python
print("array1 : {}차원".format(array1.ndim))
print("array2 : {}차원".format(array2.ndim))
print("array3 : {}차원".format(array3.ndim))
```

    array1 : 1차원
    array2 : 2차원
    array3 : 2차원
    

___

### ndarray 데이터 값 타입


```python
list1 = [1, 2, 3]
array1 = np.array(list1)

print("array1 type : {}".format(type(array1)))
print("array1 data type : {}".format(array1.dtype))

```

    array1 type : <class 'numpy.ndarray'>
    array1 data type : int32
    


```python
list2 = [1, 2, "hi"]
list3 = [3, 4, 5.5]
#list에서는 서로 다른 데이터 타입이 포합될 수 있다.

array2 = np.array(list2)
array3 = np.array(list3)
#array에는 서로 다른 데이터 타입이 포함되면 더 크기가 큰 데이터형으로 전체가 변환된다.

print(array2)
print("array2 data type : {}".format(array2.dtype))

      
print(array3)
print("array3 data type : {}".format(array3.dtype))
      
```

    ['1' '2' 'hi']
    array2 data type : <U11
    [3.  4.  5.5]
    array3 data type : float64
    

---

### astype()을 통한 타입 변환


```python
list1 = [1, 2, 3]
array1_int = np.array(list1)
print("array1_int : {}\t\tarray1_int data type : {}".format(array1_int, array1_int.dtype))

# array1_int라는 array를 float형으로 바꿔보자
# array1_int를 64bits flaot형으로 타입 변환해보도록 하자


array1_float = array1_int.astype("float64")
print("array1_float : {}\tarray1_float data type : {}".format(array1_float, array1_float.dtype))

# array1_float를 32bits int형으로 타입 변환해보도록 하자

array1_int32 = array1_float.astype("int32")
print("array1_int32 : {}\t\tarray1_int32 data type : {}".format(array1_int32, array1_int32.dtype))
```

    array1_int : [1 2 3]		array1_int data type : int32
    array1_float : [1. 2. 3.]	array1_float data type : float64
    array1_int32 : [1 2 3]		array1_int32 data type : int32
    

___

### ndarray에서 axis 기반의 연산함수 수행


```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]

array2 = np.array([list1, 
                   list2])

print(array2)

# array2에 있는 모든 원소의 합 구하기
print("array2의 모든 원소 합 : {}".format(array2.sum()))

# 행 단위로 합 구하기
print("행단위로 array2의 원소 합 : {}".format(array2.sum(axis=0)))

# 열 단위로 합 구하기
print("열단위로 array2의 원소 합 : {}".format(array2.sum(axis=1)))


```

    [[1 2 3]
     [4 5 6]]
    array2의 모든 원소 합 : 21
    행단위로 array2의 원소 합 : [5 7 9]
    열단위로 array2의 원소 합 : [ 6 15]
    


```python

```