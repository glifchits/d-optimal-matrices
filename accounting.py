from __future__ import division

import math
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


class TaskList(object):

    def __init__(self, name):
        self.name = name
        self.running_task = None
        self.task_list = []
        self.min = None
        self.max = None
        self.sum = 0

    def start_new_task(self):
        self.running_task = Task(self.name)

    def finish_current_task(self):
        assert self.has_running_task, \
            'Cannot finish task when no tasks are running'
        self.running_task.finish()
        self.task_list.append(self.running_task)
        task_time = self.running_task.total_time
        if not self.min or self.min.total_time > task_time:
            self.min = self.running_task
        if not self.max or self.max.total_time < task_time:
            self.max = self.running_task
        self.sum += task_time
        self.running_task = None

    @property
    def has_running_task(self):
        return True if self.running_task is not None else False

    @property
    def last_task(self):
        if self.has_running_task:
            return self.running_task
        else:
            return self.task_list[-1]

    def __len__(self):
        running = 1 if self.has_running_task else 0
        return len(self.task_list) + running

    def __iter__(self):
        for task in self.task_list:
            yield task

    def get_min_time(self):
        return self.min.total_time

    def get_max_time(self):
        return self.max.total_time

    def get_total_time(self):
        return self.sum

    def get_avg_time(self):
        return self.sum / len(self)

    def get_std_dev(self):
        avg = self.get_avg_time()
        _sum = 0
        _len = 0
        for task in self:
            _sum += (task.total_time - avg) ** 2
            _len += 1
        return math.sqrt(_sum / _len)


class MetaAccounting(type):

    def __getitem__(cls, task_name):
        return cls.bookkeeping[task_name]


class Accounting(object):

    __metaclass__ = MetaAccounting

    bookkeeping = {}

    @classmethod
    def start_task(cls, task_name):
        if task_name not in cls.bookkeeping:
            cls.bookkeeping[task_name] = TaskList(task_name)
        cls.bookkeeping[task_name].start_new_task()

    @classmethod
    def finish_task(cls, task_name):
        assert task_name in cls.bookkeeping, 'No task `%s`' % task_name
        task_list = cls.bookkeeping[task_name]
        task_list.finish_current_task()

    @classmethod
    def print_stats(cls):
        print "\nExecution statistics:\n"
        bk = cls.bookkeeping
        for task in sorted(bk.keys()):
            task_list = bk[task]
            print "Task %s" % task
            print "  Times executed:", len(task_list)
            print "  Min time: {0:.4f}s".format(task_list.get_min_time())
            print "  Max time: {0:.4f}s".format(task_list.get_max_time())
            print "  Avg time: {0:.4f}s".format(task_list.get_avg_time())
            print "  Total time: {0:.4f}s".format(task_list.get_total_time())
            print "   Std dev: {0:.4f}s".format(task_list.get_std_dev())
            print ""


def just_wait(seconds):
    '''
    An alternative to `time.sleep` that doesn't give up the thread so that we
    can be more precise with time-based tests.
    '''
    start = time.time()
    while start+seconds > time.time():
        pass


class TestCase(unittest.TestCase):
    # Number of digits to round to when checking time differences (in tests)
    TIME_PRECISION = 3

    def assertTimeEqual(self, a, b):
        self.assertAlmostEqual(a, b, places=self.TIME_PRECISION)


class TestTask(TestCase):

    def test_is_finished(self):
        t = Task('test')
        self.assertFalse(t.is_finished)
        t.finish()
        self.assertTrue(t.is_finished)

    def test_total_time(self):
        wait = 0.1
        t = Task('test_total_time', 0, 2)
        self.assertEqual(t.total_time, 2)

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

    def test_task_time_elapsed(self):
        t = Task('timetest')
        just_wait(0.03)
        t.finish()
        self.assertTimeEqual(t.total_time, 0.03)


class TestTaskList(TestCase):

    def test_lifecycle(self):
        tasklist = TaskList('test')
        self.assertFalse(tasklist.has_running_task)
        self.assertEqual(len(tasklist), 0)
        tasklist.start_new_task()
        self.assertTrue(tasklist.has_running_task)
        self.assertEqual(len(tasklist), 1)
        tasklist.finish_current_task()
        self.assertFalse(tasklist.has_running_task)
        self.assertEqual(len(tasklist), 1)

    def test_min_and_max(self):
        tasklist = TaskList('test2')
        tasklist.start_new_task()
        just_wait(0.1)
        tasklist.finish_current_task()
        tasklist.start_new_task()
        just_wait(0.2)
        tasklist.finish_current_task()
        # check task values
        self.assertTimeEqual(tasklist.get_min_time(), 0.1)
        self.assertTimeEqual(tasklist.get_max_time(), 0.2)
        self.assertEqual(len(tasklist), 2)

    def test_sum(self):
        tasklist = TaskList('test_sum')
        wait = 0.05
        iters = 5
        for i in range(iters):
            tasklist.start_new_task()
            just_wait(wait)
            tasklist.finish_current_task()
        self.assertEqual(len(tasklist), iters)
        self.assertEqual(
            sum(t.total_time for t in tasklist),
            tasklist.get_total_time()
        )
        self.assertTimeEqual(tasklist.get_total_time(), wait * iters)

    def test_avg(self):
        tasklist = TaskList('testavg')
        tasklist.start_new_task()
        just_wait(0.1)
        tasklist.finish_current_task()
        tasklist.start_new_task()
        just_wait(0.2)
        tasklist.finish_current_task()
        self.assertTimeEqual(tasklist.get_avg_time(), 0.15)

    def test_std_dev(self):
        tasklist = TaskList('test_stddev')
        for i in range(10):
            tasklist.start_new_task()
            just_wait(0.01 * i)
            tasklist.finish_current_task()
        self.assertEqual(len(tasklist), 10)
        self.assertTimeEqual(tasklist.get_std_dev(), 0.028722813232690138)


class TestAccounting(TestCase):

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
        task = Accounting.bookkeeping[taskname].last_task
        self.assertTrue(task.is_finished)
        self.assertTimeEqual(task.total_time, wait)

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
        self.assertTimeEqual(sum(t.total_time for t in tasks), 2 * wait)

    def test_multiple_task_types(self):
        Accounting.start_task('task1')
        just_wait(0.1)
        Accounting.start_task('task2')
        just_wait(0.2)
        Accounting.finish_task('task1')
        Accounting.finish_task('task2')
        Accounting.start_task('task1')
        Accounting.finish_task('task1')
        self.assertEqual(len(Accounting['task1']), 2)
        self.assertEqual(len(Accounting['task2']), 1)
        self.assertTimeEqual(Accounting['task1'].get_min_time(), 0)
        self.assertTimeEqual(Accounting['task1'].get_max_time(), 0.3)
        self.assertTimeEqual(Accounting['task2'].get_min_time(), 0.2)
        self.assertTimeEqual(Accounting['task2'].get_max_time(), 0.2)
        with self.assertRaises(KeyError):
            Accounting['notataskname']


if __name__ == '__main__':
    unittest.main()
