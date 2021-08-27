from photo_ocr.util.batchify import flatten


def run_flatten_and_unflatten_test(list_of_lists, expected_flat_list):
    # first check that flatten works
    actual_flat_list, unflatten = flatten(list_of_lists)
    assert expected_flat_list == actual_flat_list

    # then check that unflatten works
    actual_unflattened_list = unflatten(actual_flat_list)
    assert list_of_lists == actual_unflattened_list


# todo should not be called in photo_ocr code
def test_empty_list():
    list_of_lists = []
    expected_flat_list = []
    run_flatten_and_unflatten_test(list_of_lists, expected_flat_list)


def test_one_sublist_empty():
    list_of_lists = [[]]
    expected_flat_list = []
    run_flatten_and_unflatten_test(list_of_lists, expected_flat_list)


def test_one_sublist_one_entry():
    list_of_lists = [["test1"]]
    expected_flat_list = ["test1"]
    run_flatten_and_unflatten_test(list_of_lists, expected_flat_list)


def test_one_sublist_multiple_entries():
    list_of_lists = [["test1", "test2"]]
    expected_flat_list = ["test1", "test2"]
    run_flatten_and_unflatten_test(list_of_lists, expected_flat_list)


def test_multiple_sublists_multiple_entries():
    list_of_lists = [["test1", "test2"], [1, 4, 7], ["A", "Z"]]
    expected_flat_list = ["test1", "test2", 1, 4, 7, "A", "Z"]
    run_flatten_and_unflatten_test(list_of_lists, expected_flat_list)


def test_multiple_sublists_first_empty():
    list_of_lists = [[], [1, 4, 7], ["A", "Z"]]
    expected_flat_list = [1, 4, 7, "A", "Z"]
    run_flatten_and_unflatten_test(list_of_lists, expected_flat_list)


def test_multiple_sublists_middle_empty():
    list_of_lists = [["test1", "test2"], [], ["A", "Z"]]
    expected_flat_list = ["test1", "test2", "A", "Z"]
    run_flatten_and_unflatten_test(list_of_lists, expected_flat_list)


def test_multiple_sublists_last_empty():
    list_of_lists = [["test1", "test2"], [1, 4, 7], []]
    expected_flat_list = ["test1", "test2", 1, 4, 7]
    run_flatten_and_unflatten_test(list_of_lists, expected_flat_list)


