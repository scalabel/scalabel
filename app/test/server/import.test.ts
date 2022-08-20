import { isValidId } from "../../src/functional/states"
import { convertItemToImport } from "../../src/server/import"
import { sampleStateExportImagePolygonMulti } from "../test_states/test_export_objects"

describe("test import functionality for polygon", () => {
  test("import multi-component polygons", () => {
    const item = convertItemToImport(
      "demo",
      0,
      {
        "-1": sampleStateExportImagePolygonMulti[0]
      },
      0,
      0,
      {},
      {},
      { unlabeled: 0 },
      true,
      ["polygon2d"]
    )
    const labels = Object.entries(item.labels).map(([_id, l]) => l)
    expect(labels).toHaveLength(3)

    const root = labels.find((l) => !isValidId(l.parent))
    expect(root).toBeTruthy()

    labels.forEach((l) => {
      if (l.id === root?.id) {
        return
      }
      expect(l.parent).toEqual(root?.id)
    })
  })
})
