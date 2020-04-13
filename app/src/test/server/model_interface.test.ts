import { LabelTypeName } from '../../js/common/types'
import { PathPoint2D, PointType } from '../../js/drawable/2d/path_point2d'
import { makePolygon } from '../../js/functional/states'
import { PolygonType, RectType } from '../../js/functional/types'
import { convertPolygonToExport } from '../../js/server/export'
import { ModelInterface } from '../../js/server/model_interface'
import { ModelEndpoint } from '../../js/server/types'
import { checkPathPointFields } from '../util'

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
    const rect: RectType = {
      x1: 5, y1: 2, x2: 6, y2: 10
    }
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
      (new PathPoint2D(0, 1, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(5, 3, PointType.VERTEX)).toPathPoint()
    ]
    const poly2d = makePolygon({ points })
    const itemIndex = 0
    const labelType = LabelTypeName.POLYGON_2D
    const query = modelInterface.makePolyQuery(
      poly2d, url, itemIndex, labelType)
    expect(query.endpoint).toBe(ModelEndpoint.REFINE_POLY)
    expect(query.itemIndex).toBe(itemIndex)

    const itemData = query.data
    expect(itemData.name).toBe(projectName)
    expect(itemData.url).toBe(url)

    const expectedPoly = convertPolygonToExport(poly2d, labelType)
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

    const shape = action.shapes[0][0][0] as PolygonType
    const points = shape.points
    checkPathPointFields(points[0], 1, 5, true)
    checkPathPointFields(points[1], 100, -5, true)
  })
})
