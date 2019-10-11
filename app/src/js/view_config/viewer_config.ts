import * as THREE from 'three'
import { updateImageViewerConfig, zoomImage } from '../action/image'
import { dragCamera, moveBack, moveCameraAndTarget, moveDown, moveForward, moveLeft, moveRight, moveUp, rotateCamera, zoomCamera } from '../action/point_cloud'
import Session from '../common/session'
import { Key } from '../common/types'
import { getCurrentImageViewerConfig, getCurrentItem, getCurrentPointCloudViewerConfig } from '../functional/state_util'
import { Vector3D } from '../math/vector3d'
import { SCROLL_ZOOM_RATIO } from './image'

/**
 * Static class for updating viewer config in response to UI events
 */

/**
 * Normalize mouse coordinates to make canvas left top origin
 * @param x
 * @param y
 * @param canvas
 */
function normalizeCoordinatesToCanvas (
  x: number, y: number, container: HTMLDivElement): number[] {
  return [
    x - container.getBoundingClientRect().left,
    y - container.getBoundingClientRect().top
  ]
}

/**
 * Class for managing viewer config
 */
export default class ViewerConfigUpdater {
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** Mouse x-coord */
  private _mX: number
  /** Mouse y-coord */
  private _mY: number
  /** Whether mouse is down */
  private _mouseDown: boolean
  /** which button is pressed on mouse down */
  private _mouseButton: number
  /** Display */
  private _container: HTMLDivElement | null
  /** Camera for math */
  private _camera: THREE.PerspectiveCamera
  /** Raycaster */
  private _raycaster: THREE.Raycaster
  /** target */
  private _target: THREE.Vector3

  constructor () {
    this._keyDownMap = {}
    this._mX = 0
    this._mY = 0
    this._container = null
    this._mouseDown = false
    this._camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
    this._raycaster = new THREE.Raycaster()
    this._target = new THREE.Vector3()
    this._mouseButton = 0

    document.addEventListener('contextmenu', (e) => {
      e.preventDefault()
    }, false)
  }

  /**
   * Set container
   * @param container
   */
  public setContainer (container: HTMLDivElement) {
    this._container = container
  }

  /**
   * Update state
   * @param state
   */
  public updateCamera () {
    const state = Session.getState()
    if (this._container) {
      this._camera.aspect = this._container.offsetWidth /
        this._container.offsetHeight
      this._camera.updateProjectionMatrix()
    }
    const config = getCurrentPointCloudViewerConfig(state)
    this._target.x = config.target.x
    this._target.y = config.target.y
    this._target.z = config.target.z

    this._camera.up.x = config.verticalAxis.x
    this._camera.up.y = config.verticalAxis.y
    this._camera.up.z = config.verticalAxis.z
    this._camera.position.x = config.position.x
    this._camera.position.y = config.position.y
    this._camera.position.z = config.position.z
    this._camera.lookAt(this._target)
  }

  /**
   * Handle mouse movement
   * @param e
   */
  public onMouseMove (e: MouseEvent) {
    if (!this._container) {
      return
    }

    const state = Session.getState()
    const normalized = normalizeCoordinatesToCanvas(
      e.clientX, e.clientY, this._container
    )
    if (this._mouseDown) {
      switch (state.task.config.itemType) {
        case 'image':
          if (this.isKeyDown('Control')) {
            const dx = normalized[0] - this._mX
            const dy = normalized[1] - this._mY
            const displayLeft = this._container.scrollLeft - dx
            const displayTop = this._container.scrollTop - dy
            Session.dispatch(updateImageViewerConfig({
              displayLeft, displayTop
            }))
          }
          break
        case 'pointcloud':
          const viewerConfig = getCurrentPointCloudViewerConfig(state)
          if (this._mouseButton === 2) {
            this.updateCamera()
            Session.dispatch(dragCamera(
              this._mX,
              this._mY,
              normalized[0],
              normalized[1],
              this._camera,
              viewerConfig
            ))
          } else {
            Session.dispatch(rotateCamera(
              this._mX,
              this._mY,
              normalized[0],
              normalized[1],
              viewerConfig
            ))
          }
          break
      }
    }
    this._mX = normalized[0]
    this._mY = normalized[1]
  }

  /**
   * Handle mouse down
   * @param e
   */
  public onMouseDown (e: MouseEvent) {
    if (!this._container) {
      return
    }
    this._mouseDown = true
    this._mouseButton = e.button
    if (this._mouseButton === 2) {
      e.preventDefault()
    }
    const normalized = normalizeCoordinatesToCanvas(
      e.clientX, e.clientY, this._container
    )
    this._mX = normalized[0]
    this._mY = normalized[1]
  }

  /**
   * Handle mouse up
   * @param e
   */
  public onMouseUp (_e: MouseEvent) {
    this._mouseDown = false
  }

  /**
   * Handle double click
   * @param e
   */
  public onDoubleClick (e: MouseEvent) {
    if (!this._container) {
      return
    }

    const state = Session.getState()
    const normalized = normalizeCoordinatesToCanvas(
      e.clientX, e.clientY, this._container
    )
    switch (state.task.config.itemType) {
      case 'image':
        break
      case 'pointcloud':
        this.updateCamera()

        const NDC = this.convertMouseToNDC(
          normalized[0],
          normalized[1]
        )
        const x = NDC[0]
        const y = NDC[1]

        this._raycaster.linePrecision = 0.2
        this._raycaster.setFromCamera(new THREE.Vector2(x, y), this._camera)
        const item = getCurrentItem(state)
        const pointCloud = Session.pointClouds[item.index]

        const intersects = this._raycaster.intersectObject(pointCloud)

        if (intersects.length > 0) {
          const newTarget = intersects[0].point
          const viewerConfig = getCurrentPointCloudViewerConfig(state)
          Session.dispatch(moveCameraAndTarget(
            new Vector3D(
              viewerConfig.position.x - viewerConfig.target.x + newTarget.x,
              viewerConfig.position.y - viewerConfig.target.y + newTarget.y,
              viewerConfig.position.z - viewerConfig.target.z + newTarget.z
            ),
            new Vector3D(
              newTarget.x,
              newTarget.y,
              newTarget.z
            )
          ))
        }
        break
    }
  }

  /**
   * Handle mouse wheel
   * @param _e
   */
  public onWheel (e: WheelEvent) {
    const state = Session.getState()
    switch (state.task.config.itemType) {
      case 'image':
        if (this.isKeyDown('Control')) { // control for zoom
          e.preventDefault()
          let zoomRatio = SCROLL_ZOOM_RATIO
          if (e.deltaY < 0) {
            zoomRatio = 1. / zoomRatio
          }
          const imageZoomAction = zoomImage(
            zoomRatio,
            getCurrentImageViewerConfig(state)
          )
          if (imageZoomAction) {
            Session.dispatch(imageZoomAction)
          }
        }
        break
      case 'pointcloud':
        const pointCloudZoomAction = zoomCamera(
          getCurrentPointCloudViewerConfig(state), e.deltaY
        )
        if (pointCloudZoomAction) {
          Session.dispatch(pointCloudZoomAction)
        }
        break
    }
  }

  /**
   * Handle key down
   * @param e
   */
  public onKeyDown (e: KeyboardEvent) {
    const key = e.key
    this._keyDownMap[key] = true
    const state = Session.getState()
    switch (state.task.config.itemType) {
      case 'image':
        break
      case 'pointcloud':
        this.pointCloudKeyEvents(e.key)
        break
    }
  }

  /**
   * Handle key up
   * @param e
   */
  public onKeyUp (e: KeyboardEvent) {
    const key = e.key
    delete this._keyDownMap[key]
  }

  /**
   * Point cloud viewer config key events
   * @param key
   */
  private pointCloudKeyEvents (key: string) {
    const viewerConfig = getCurrentPointCloudViewerConfig(Session.getState())
    switch (key) {
      case Key.PERIOD:
        Session.dispatch(moveUp(viewerConfig))
        break
      case Key.SLASH:
        Session.dispatch(moveDown(viewerConfig))
        break
      case Key.DOWN:
      case Key.ARROW_DOWN:
        Session.dispatch(moveBack(viewerConfig))
        break
      case Key.UP:
      case Key.ARROW_UP:
        Session.dispatch(moveForward(viewerConfig))
        break
      case Key.LEFT:
      case Key.ARROW_LEFT:
        Session.dispatch(moveLeft(viewerConfig))
        break
      case Key.RIGHT:
      case Key.ARROW_RIGHT:
        Session.dispatch(moveRight(viewerConfig))
        break
    }
  }

  /**
   * Whether a specific key is pressed down
   * @param {string} key - the key to check
   * @return {boolean}
   */
  private isKeyDown (key: string): boolean {
    return this._keyDownMap[key]
  }

  /**
   * Normalize coordinates to display
   * @param mX
   * @param mY
   */
  private convertMouseToNDC (mX: number, mY: number) {
    if (this._container) {
      let x = mX / this._container.offsetWidth
      let y = mY / this._container.offsetHeight
      x = 2 * x - 1
      y = -2 * y + 1

      return [x, y]
    }
    return [0, 0]
  }
}
