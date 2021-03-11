import { AttributeToolType } from "../../src/const/common"
import * as makers from "../../src/functional/states"
import * as stats from "../../src/server/stats"
import { TaskType } from "../../src/types/state"

let sampleTask1: TaskType
let sampleTask2: TaskType
let allTasks: TaskType[]
let expectedLabelStats: stats.LabelStats

beforeAll(() => {
  const config = makers.makeTaskConfig({
    categories: ["car", "person", "traffic light"],
    attributes: [
      makers.makeAttribute({
        name: "Occluded"
      }),
      makers.makeAttribute({
        name: "Traffic light color",
        toolType: AttributeToolType.LIST,
        values: ["N/A", "G", "Y", "R"]
      })
    ]
  })

  sampleTask1 = makers.makeTask({
    config,
    items: [
      makers.makeItem({
        labels: {
          0: makers.makeLabel({
            category: [1],
            attributes: {
              0: [1],
              1: [2]
            }
          }),
          1: makers.makeLabel({
            category: [0],
            attributes: {
              0: [0],
              1: [3]
            }
          })
        }
      }),
      makers.makeItem(),
      makers.makeItem({
        labels: {
          2: makers.makeLabel({
            category: [1],
            attributes: {
              0: [1],
              1: [0]
            }
          }),
          3: makers.makeLabel({
            category: [2],
            attributes: {
              0: [1],
              1: [1]
            }
          })
        }
      })
    ],
    progress: {
      submissions: [{ time: 55, user: "sampleUser " }]
    }
  })

  sampleTask2 = makers.makeTask({
    config,
    items: [
      makers.makeItem(),
      makers.makeItem({
        labels: {
          4: makers.makeLabel(),
          5: makers.makeLabel({
            category: [1],
            attributes: {
              0: [1],
              1: [2]
            }
          }),
          6: makers.makeLabel({
            category: [2, 0],
            attributes: {
              0: [0],
              1: [3, 2]
            }
          })
        }
      })
    ]
  })

  const itemlessTask = makers.makeTask({ config })

  allTasks = [sampleTask1, sampleTask2, itemlessTask]
  expectedLabelStats = {
    category: {
      car: {
        count: 2,
        attribute: {
          Occluded: { false: 2, true: 0 },
          "Traffic light color": { "N/A": 0, G: 0, Y: 1, R: 2 }
        }
      },
      person: {
        count: 3,
        attribute: {
          Occluded: { false: 0, true: 3 },
          "Traffic light color": { "N/A": 1, G: 0, Y: 2, R: 0 }
        }
      },
      "traffic light": {
        count: 2,
        attribute: {
          Occluded: { false: 1, true: 1 },
          "Traffic light color": { "N/A": 0, G: 1, Y: 1, R: 1 }
        }
      }
    },
    attribute: {
      Occluded: { false: 2, true: 4 },
      "Traffic light color": { "N/A": 1, G: 1, Y: 3, R: 2 }
    }
  }
})

describe("Simple stat functions", () => {
  test("Count labels", () => {
    expect(stats.countLabelsTask(sampleTask1)).toBe(4)
    expect(stats.countLabelsTask(sampleTask2)).toBe(3)
    expect(stats.countLabelsProject(allTasks)).toBe(7)
  })

  test("Count labeled images", () => {
    expect(stats.countLabeledItemsTask(sampleTask1)).toBe(2)
    expect(stats.countLabeledItemsTask(sampleTask2)).toBe(1)
    expect(stats.countLabeledItemsProject(allTasks)).toBe(3)
  })

  test("Count items", () => {
    expect(stats.getNumItems(allTasks)).toBe(5)
  })

  test("Count submissions", () => {
    expect(stats.getNumSubmissions(allTasks))
  })

  test("Category counts", () => {
    expect(stats.getLabelStats(allTasks)).toStrictEqual(expectedLabelStats)
  })
})

describe("Stat aggregation", () => {
  const constantDate = Date.now()
  const dateFn = Date.now

  beforeAll(() => {
    Date.now = jest.fn(() => {
      return constantDate
    })
  })

  afterAll(() => {
    Date.now = dateFn
  })

  test("Task options", () => {
    expect(stats.getTaskOptions(sampleTask1)).toStrictEqual({
      numLabeledItems: "2",
      numLabels: "4",
      submissions: [{ time: 55, user: "sampleUser " }],
      handlerUrl: ""
    })
  })

  test("Project stats", () => {
    expect(stats.getProjectStats(allTasks)).toStrictEqual({
      numLabels: 7,
      numLabeledItems: 3,
      numItems: 5,
      numSubmittedTasks: 1,
      numTasks: 3,
      labelStats: expectedLabelStats,
      timestamp: constantDate
    })
  })

  test("Project stats empty task list", () => {
    expect(stats.getProjectStats([])).toStrictEqual({
      numLabels: 0,
      numLabeledItems: 0,
      numItems: 0,
      numSubmittedTasks: 0,
      numTasks: 0,
      labelStats: { category: {}, attribute: {} },
      timestamp: constantDate
    })
  })
})
