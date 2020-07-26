import { LabelTypeName } from '../../js/common/types'
import { ModelEndpoint } from '../../js/const/connection'
import { makePathPoint2D, makeRect } from '../../js/functional/states'
import { PathPoint2DType, PathPointType, RectType } from '../../js/functional/types'
import { convertPolygonToExport } from '../../js/server/export'
import { ModelInterface } from '../../js/server/model_interface'

let modelInterface: ModelInterface
let projectName: string
let sessionId: string
let url: string

beforeAll(() => {
  projectName = 'projectName'
  sessionId = 'sessionId'
  url = 'testurl'
  modelInterface = new ModelInterface(projectName, sessionId)
})

describe('test model interface query construction', () => {
  test('rect query construction', () => {
    const rect: RectType = makeRect({
      x1: 5, y1: 2, x2: 6, y2: 10
    })
    const itemIndex = 1
    const query = modelInterface.makeRectQuery(rect, url, itemIndex)
    expect(query.endpoint).toBe(ModelEndpoint.PREDICT_POLY)
    expect(query.itemIndex).toBe(itemIndex)

    const itemData = query.data
    expect(itemData.name).toBe(projectName)
    expect(itemData.url).toBe(url)
    expect(itemData.labels[0].box2d).toEqual(rect)
  })

  test('poly query construction', () => {
    const points = [
      makePathPoint2D({ x: 0, y: 1, pointType: PathPointType.LINE }),
      makePathPoint2D({ x: 5, y: 3, pointType: PathPointType.LINE })
    ]
    const itemIndex = 0
    const labelType = LabelTypeName.POLYGON_2D
    const query = modelInterface.makePolyQuery(
      points, url, itemIndex, labelType)
    expect(query.endpoint).toBe(ModelEndpoint.REFINE_POLY)
    expect(query.itemIndex).toBe(itemIndex)

    const itemData = query.data
    expect(itemData.name).toBe(projectName)
    expect(itemData.url).toBe(url)

    const expectedPoly = convertPolygonToExport(points, labelType)
    expect(itemData.labels[0].poly2d).toEqual(expectedPoly)
  })
})

describe('test model interface action translation', () => {
  test('poly action translation', () => {
    const polyPoints = [[1, 5], [100, -5]]
    const itemIndex = 3
    const action = modelInterface.makePolyAction(polyPoints, itemIndex)
    expect(action.sessionId).toBe(sessionId)

    const label = action.labels[0][0]
    expect(label.manual).toBe(false)

    const points = action.shapes[0][0] as PathPoint2DType[]
    expect(points[0]).toMatchObject({ x: 1, y: 5, pointType: 'line' })
    expect(points[1]).toMatchObject({ x: 100, y: - 5, pointType: 'line' })
  })
})
