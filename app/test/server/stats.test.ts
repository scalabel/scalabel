import { makeItem, makeLabel, makeTask } from '../../src/functional/states'
import { countLabels } from '../../src/server/stats'

test('Count labels', () => {
  const sampleTask = makeTask()
  sampleTask.items = [
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
  const [numLabeledItems, numLabels] = countLabels(sampleTask)
  expect(numLabeledItems).toBe(2)
  expect(numLabels).toBe(4)

})
