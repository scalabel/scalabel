import {
  makeItemExport,
  makeLabel,
  makeLabelExport
} from "../../src/functional/states"
import { ItemExport, LabelExport } from "../../src/types/export"

describe("Defaults get overwritten", () => {
  test("Item export creation", () => {
    const name = "itemName"
    const timestamp = 5
    const partialItemExport: Partial<ItemExport> = {
      name,
      timestamp
    }
    const itemExport = makeItemExport(partialItemExport)
    expect(itemExport.name).toBe(name)
    expect(itemExport.timestamp).toBe(timestamp)
    expect(itemExport.url).toBe("")
  })

  test("Label export creation", () => {
    const label = makeLabel()
    const category = "category"
    const partialLabelExport: Partial<LabelExport> = {
      id: label.id,
      category
    }
    const labelExport = makeLabelExport(partialLabelExport)
    expect(labelExport.id).toBe(label.id)
    expect(labelExport.category).toBe(category)
    expect(labelExport.box2d).toBe(null)
  })

  test("Attributes are immutable", () => {
    const attributes = {
      0: [1],
      1: [0]
    }
    const label = makeLabel({
      attributes
    })
    expect(label.attributes[0]).toStrictEqual([1])
    expect(label.attributes[1]).toStrictEqual([0])

    // Mutating original attribute should not affect the label
    attributes[0] = [0]
    expect(label.attributes[0]).toStrictEqual([1])
    expect(label.attributes[1]).toStrictEqual([0])
  })
})
