import { addBox2dLabel } from "../action/box2d"
import { addBox3dLabel } from "../action/box3d"
import { addPolygon2dLabel } from "../action/polygon2d"
import { ModelEndpoint } from "../const/connection"
import {
  makeItemExport,
  makeLabelExport,
  makeSimplePathPoint2D
} from "../functional/states"
import { Vector3D } from "../math/vector3d"
import { AddLabelsAction } from "../types/action"
import { ModelQuery, ModelRequest } from "../types/message"
import {
  IntrinsicsType,
  PathPoint2DType,
  PathPointType,
  RectType,
  SimpleRect
} from "../types/state"
import { intrinsicsToExport } from "./bdd_type_transformers"
import { convertPolygonToExport } from "./export"
import { ItemExport } from "../types/export"

/**
 * API between redux style data and data for the models
 */
export class ModelInterface {
  /** project name */
  public projectName: string
  /** current session id */
  public sessionId: string

  /**
   * Constructor
   *
   * @param projectName
   * @param sessionId
   */
  constructor(projectName: string, sessionId: string) {
    this.projectName = projectName
    this.sessionId = sessionId
  }

  /**
   * Query for 'rect -> polygon' segmentation
   *
   * @param rect
   * @param url
   * @param itemIndex
   */
  public makeRectRequest(
    rect: RectType,
    url: string,
    itemIndex: number
  ): ModelRequest {
    const label = makeLabelExport({
      box2d: rect
    })
    const item = makeItemExport({
      name: this.projectName,
      url,
      labels: [label]
    })
    return {
      data: [item],
      itemIndices: [itemIndex]
    }
  }

  /**
   * Query for refining 'polygon -> polygon' segmentation
   *
   * @param points
   * @param url
   * @param itemIndex
   * @param labelType
   */
  public makePolyQuery(
    points: PathPoint2DType[],
    url: string,
    itemIndex: number,
    labelType: string
  ): ModelQuery {
    const poly2d = convertPolygonToExport(points, labelType)
    const label = makeLabelExport({
      poly2d
    })
    const item = makeItemExport({
      name: this.projectName,
      url,
      labels: [label]
    })
    return {
      data: item,
      endpoint: ModelEndpoint.REFINE_POLY,
      itemIndex
    }
  }

  /**
   * Request for image prediction
   *
   * @param urls
   * @param itemIndices
   * @param intrinsics
   */
  public makeImageRequest(
    urls: string[],
    itemIndices: number[],
    intrinsics: Array<IntrinsicsType | undefined>
  ): ModelRequest {
    const items: ItemExport[] = []
    for (let index = 0; index < urls.length; index++) {
      const item = makeItemExport({
        name: this.projectName,
        url: urls[index],
        intrinsics:
          intrinsics[index] !== null && intrinsics[index] !== undefined
            ? intrinsicsToExport(intrinsics[index] as IntrinsicsType)
            : undefined
      })
      items.push(item)
    }
    return {
      data: items,
      itemIndices
    }
  }

  /**
   * Translate box prediction response to an action
   *
   * @param predictedBox
   * @param itemIndex
   */
  public makeRectAction(
    predictedBox: number[],
    itemIndex: number
  ): AddLabelsAction {
    const box: SimpleRect = {
      x1: predictedBox[0],
      y1: predictedBox[1],
      x2: predictedBox[2],
      y2: predictedBox[3]
    }
    const action = addBox2dLabel(itemIndex, -1, [0], {}, box, false)
    action.sessionId = this.sessionId
    return action
  }

  /**
   * Translate polygon response to an action
   *
   * @param polyPoints
   * @param itemIndex
   */
  public makePolyAction(
    polyPoints: number[][],
    itemIndex: number
  ): AddLabelsAction {
    const points = polyPoints.map((point: number[]) => {
      return makeSimplePathPoint2D(point[0], point[1], PathPointType.LINE)
    })

    const action = addPolygon2dLabel(itemIndex, -1, [0], points, true, false)
    action.sessionId = this.sessionId
    return action
  }

  /**
   * Translate box3d response to an action
   *
   * @param box3d
   * @param itemIndex
   */
  public makeBox3dAction(
    box3d: number[], // W, L, H, x, y, z, rot_y, alpha
    itemIndex: number
  ): AddLabelsAction {
    const size = new Vector3D(box3d[0], box3d[1], box3d[2]).toState()
    const center = new Vector3D(box3d[3], box3d[4], box3d[5]).toState()
    const orientation = new Vector3D(box3d[6], box3d[7], box3d[8]).toState()
    const action = addBox3dLabel(
      itemIndex,
      [-1],
      [0],
      center,
      size,
      orientation
    )
    action.sessionId = this.sessionId
    return action
  }
}
