import * as THREE from 'three'
import { IdType, INVALID_ID } from '../../functional/types'

/**
 * Basic class for object 3d so that we can assign the ids
 */
export abstract class Object3D extends THREE.Object3D {
    /** id */
  protected _shapeId: IdType

  constructor () {
    super()
    this._shapeId = INVALID_ID
  }
}
