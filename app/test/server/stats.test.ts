import { makeItem, makeLabel, makeTask } from '../../src/functional/states'
import * as stats from '../../src/server/stats'
import { TaskType } from '../../src/types/state'

let sampleTask1: TaskType
let sampleTask2: TaskType
let allTasks: TaskType[]

beforeAll(() => {
  sampleTask1 = makeTask()
  sampleTask1.items = [
    makeItem({
      labels: {
        0: makeLabel(),
        1: makeLabel()
      }}),
    makeItem(),
    makeItem({
      labels: {
        2: makeLabel(),
        3: makeLabel()
      }
    })
  ]
  sampleTask1.progress = {
    submissions: [
      { time: 55, user: 'sampleUser ' }
    ]
  }

  sampleTask2 = makeTask()
  sampleTask2.items = [
    makeItem(),
    makeItem({
      labels: {
        4: makeLabel(),
        5: makeLabel(),
        6: makeLabel()
      }
    })
  ]

  allTasks = [sampleTask1, sampleTask2]
})

describe('Simple stat functions', () => {
  test('Count labels', () => {
    expect(stats.countLabelsTask(sampleTask1)).toBe(4)
    expect(stats.countLabelsTask(sampleTask2)).toBe(3)
    expect(stats.countLabelsProject(allTasks)).toBe(7)
  })

  test('Count labeled images', () => {
    expect(stats.countLabeledItemsTask(sampleTask1)).toBe(2)
    expect(stats.countLabeledItemsTask(sampleTask2)).toBe(1)
    expect(stats.countLabeledItemsProject(allTasks)).toBe(3)
  })

  test('Count items', () => {
    expect(stats.getNumItems(allTasks)).toBe(5)
  })

  test('Count submissions', () => {
    expect(stats.getNumSubmissions(allTasks))
  })

  test('Category counts', () => {
    expect(stats.getCategoryCounts(allTasks)).toStrictEqual({
      a: 2
    })
  })
})

describe('Stat aggregation', () => {
  test('Task options', () => {
    expect(stats.getTaskOptions(sampleTask1)).toStrictEqual({
      numLabeledItems: '2',
      numLabels: '4',
      submissions: [
        { time: 55, user: 'sampleUser ' }
      ],
      handlerUrl: ''
    })
  })

  test('Project stats', () => {
    expect(stats.getProjectStats(allTasks)).toStrictEqual({
      numLabels: 7,
      numLabeledItems: 3,
      numItems: 5,
      numSubmittedTasks: 1,
      numTasks: 2,
      categoryCounts: {
        a: 2
      }
    })
  })
})
