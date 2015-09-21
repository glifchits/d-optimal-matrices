from __future__ import division

import time
import unittest


class Task(object):

    def __init__(self, name, start_time=None, end_time=None):
        self.name = name
        self.start_time = time.time() if start_time is None else start_time
        self.end_time = end_time

    def finish(self):
        self.end_time = time.time()

    @property
    def is_finished(self):
        return self.end_time != None

    @property
    def total_time(self):
        if not self.is_finished:
            return float('inf')
        return self.end_time - self.start_time

    def assert_same_type(self, other):
        assert self.name == other.name, "Can't compare %s and %s" % (
            self.name, other.name
        )

    def __lt__(self, other):
        self.assert_same_type(other)
        this = self.total_time
        that = other.total_time
        return this < that

    def __eq__(self, other):
        return self.name == other.name and \
            self.start_time == other.start_time and \
            self.end_time == other.end_time

    def __add__(self, other):
        self.assert_same_type(other)
        return Task(self.name, 0, self.total_time + other.total_time)

    def __str__(self):
        return "Task {0} ({1}) {2}".format(
            self.name, self.is_finished, self.total_time
        )


class Accounting(object):

    bookkeeping = {}

    @classmethod
    def start_task(cls, task_name):
        if task_name not in cls.bookkeeping:
            cls.bookkeeping[task_name] = []
        new_task = Task(task_name)
        cls.bookkeeping[task_name].append(new_task)

    @classmethod
    def finish_task(cls, task_name):
        assert task_name in cls.bookkeeping, 'No task `%s`' % task_name
        task_list = cls.bookkeeping[task_name]
        last_task = task_list[-1]
        assert not last_task.is_finished, 'No running task `%s`' % task_name
        last_task.finish()

    @classmethod
    def print_stats(cls):
        print "\nExecution statistics:\n"
        bk = cls.bookkeeping
        for task in sorted(bk.keys()):
            task_list = bk[task]
            sum_task_list = sum(t.total_time for t in task_list)
            print "Task %s" % task
            print "  Times executed:", len(task_list)
            print "  Min time: {0:.4f}s".format(min(task_list).total_time)
            print "  Max time: {0:.4f}s".format(max(task_list).total_time)
            print "  Avg time: {0:.4f}s".format(sum_task_list / len(task_list))
            print "  Total time spent: {0:.4f}s".format(sum_task_list)
            print ""


# Number of digits to round to when checking time differences (in tests)
TIME_PRECISION = 3

def just_wait(seconds):
    '''
    An alternative to `time.sleep` that doesn't give up the thread so that we
    can be more precise with time-based tests.
    '''
    start = time.time()
    while start+seconds > time.time():
        pass


class TestTask(unittest.TestCase):

    def test_is_finished(self):
        t = Task('test')
        self.assertFalse(t.is_finished)
        t.finish()
        self.assertTrue(t.is_finished)

    def test_total_time(self):
        wait = 0.1
        t = Task('test_total_time', 0, 2)
        self.assertAlmostEqual(t.total_time, 2)

    def test_add(self):
        t1 = Task('test1', 0, 1)
        t2 = Task('test1', 3, 4)
        t3 = t1 + t2
        self.assertEqual(t3.total_time, 2)

    def test_add_floats(self):
        t1 = Task('test', 0.001, 0.123)
        t2 = Task('test', 0.333, 0.444)
        t3 = t1 + t2
        self.assertEqual(t3.total_time, 0.122+0.111)

    def test_comparison_fail(self):
        t1 = Task('type1', 0, 1)
        t2 = Task('type2', 1, 3)
        with self.assertRaises(AssertionError):
            t1 < t2

    def test_comparison(self):
        taskname = 'test1'
        t1 = Task(taskname, 0, 1)
        t2 = Task(taskname, 0, 5)
        self.assertTrue(t1 < t2)
        self.assertFalse(t2 < t1)
        self.assertFalse(t1 == t2)


class TestAccounting(unittest.TestCase):

    def setUp(self):
        Accounting.bookkeeping = {}

    def test_task_lifecycle(self):
        # test parameters
        taskname = 'test1'
        wait = 0.1
        # start test
        Accounting.start_task(taskname)
        just_wait(wait)
        Accounting.finish_task(taskname)
        task = Accounting.bookkeeping[taskname][0]
        self.assertTrue(task.is_finished)
        self.assertAlmostEqual(task.total_time, wait, places=TIME_PRECISION)

    def test_multiple_task_stats(self):
        # test params
        taskname = 'test2'
        wait = 0.1
        # start test
        Accounting.start_task(taskname)
        just_wait(wait)
        Accounting.finish_task(taskname)
        Accounting.start_task(taskname)
        just_wait(wait)
        Accounting.finish_task(taskname)
        # check
        tasks = Accounting.bookkeeping[taskname]
        self.assertEqual(len(tasks), 2)
        self.assertAlmostEqual(sum(t.total_time for t in tasks), 2 * wait,
                places=TIME_PRECISION)


if __name__ == '__main__':
    unittest.main()
