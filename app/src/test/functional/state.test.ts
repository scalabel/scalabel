import { ItemExport, LabelExport } from '../../js/functional/bdd_types'
import { makeItemExport, makeLabelExport } from '../../js/functional/states'

describe('Defaults get overwritten', () => {
  test('Item export creation',() => {
    const name = 'itemName'
    const timestamp = 5
    const partialItemExport: Partial<ItemExport> = {
      name, timestamp
    }
    const itemExport = makeItemExport(partialItemExport)
    expect(itemExport.name).toBe(name)
    expect(itemExport.timestamp).toBe(timestamp)
    expect(itemExport.url).toBe('')
  })
  test('Label export creation',() => {
    const id = 20
    const category = 'category'
    const partialLabelExport: Partial<LabelExport> = {
      id, category
    }
    const labelExport = makeLabelExport(partialLabelExport)
    expect(labelExport.id).toBe(id)
    expect(labelExport.category).toBe(category)
    expect(labelExport.box2d).toBe(null)
  })
})
