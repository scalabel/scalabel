import { Label2DTemplateType } from '../../functional/types'
import { LabelTypeName } from '../types'
import { Box2DTrack } from './box2d_track'
import { Box3DTrack } from './box3d_track'
import { CustomLabel2DTrack } from './custom_label_track'
import { Plane3DTrack } from './plane3d_track'
import { PolygonTrack } from './polygon_track'

/**
 * Make track
 */
export function trackFactory (
  labelType: string,
  labelTemplates: { [name: string]: Label2DTemplateType }
) {
  switch (labelType) {
    case LabelTypeName.BOX_2D:
      return new Box2DTrack()
    case LabelTypeName.POLYGON_2D:
    case LabelTypeName.POLYLINE_2D:
      return new PolygonTrack()
    case LabelTypeName.BOX_3D:
      return new Box3DTrack()
    case LabelTypeName.PLANE_3D:
      return new Plane3DTrack()
    default:
      if (labelType in labelTemplates) {
        return new CustomLabel2DTrack()
      }
  }
  return null
}
