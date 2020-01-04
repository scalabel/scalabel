import createStyles from '@material-ui/core/styles/createStyles'
import { withStyles } from '@material-ui/core/styles/index'
import * as React from 'react'
import * as THREE from 'three'
import Session from '../common/session'
import { ViewerConfigTypeName } from '../common/types'
import { Label3DHandler } from '../drawable/3d/label3d_handler'
import { isCurrentFrameLoaded } from '../functional/state_util'
import { Image3DViewerConfigType, State } from '../functional/types'
import { MAX_SCALE, MIN_SCALE, updateCanvasScale } from '../view_config/image'
import { convertMouseToNDC } from '../view_config/point_cloud'
import { DrawableCanvas } from './viewer'

const styles = () => createStyles({
  label3d_canvas: {
    position: 'absolute',
    height: '100%',
    width: '100%'
  }
})

interface ClassType {
  /** CSS canvas name */
  label3d_canvas: string
}

interface Props {
  /** CSS class */
  classes: ClassType
  /** container */
  display: HTMLDivElement | null
  /** viewer id */
  id: number
  /** camera */
  camera: THREE.Camera
}

/**
 * Normalize mouse coordinates to make canvas left top origin
 * @param x
 * @param y
 * @param canvas
 */
function normalizeCoordinatesToCanvas (
  x: number, y: number, canvas: HTMLCanvasElement): number[] {
  return [
    x - canvas.getBoundingClientRect().left,
    y - canvas.getBoundingClientRect().top
  ]
}

/**
 * Canvas Viewer
 */
export class Label3dCanvas extends DrawableCanvas<Props> {
  /** Canvas to draw on */
  private canvas: HTMLCanvasElement | null
  /** Container */
  private display: HTMLDivElement | null
  /** Current scale */
  private scale: number
  /** ThreeJS Renderer */
  private renderer?: THREE.WebGLRenderer
  /** ThreeJS Camera */
  private camera: THREE.Camera
  /** raycaster */
  private _raycaster: THREE.Raycaster
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** Flag set if data is 2d */
  private data2d: boolean

  /** drawable label list */
  private _labelHandler: Label3DHandler

  /** key up listener */
  private _keyUpListener: (e: KeyboardEvent) => void
  /** key down listener */
  private _keyDownListener: (e: KeyboardEvent) => void
  /** drawable callback */
  private _drawableUpdateCallback: () => void

  /**
   * Constructor, ons subscription to store
   * @param {Object} props: react props
   */
  constructor (props: Readonly<Props>) {
    super(props)
    this.camera = props.camera

    this._labelHandler = new Label3DHandler(this.camera)

    this.display = null
    this.canvas = null
    this.scale = 1
    this.data2d = false

    this._raycaster = new THREE.Raycaster()
    this._raycaster.near = 1.0
    this._raycaster.far = 100.0
    this._raycaster.linePrecision = 0.02

    this._keyDownMap = {}

    this._keyUpListener = (e) => { this.onKeyUp(e) }
    this._keyDownListener = (e) => { this.onKeyDown(e) }
    this._drawableUpdateCallback = this.renderThree.bind(this)
  }

  /**
   * Mount callback
   */
  public componentDidMount () {
    super.componentDidMount()
    document.addEventListener('keydown', this._keyDownListener)
    document.addEventListener('keyup', this._keyUpListener)
    Session.label3dList.subscribe(this._drawableUpdateCallback)
  }

  /**
   * Unmount callback
   */
  public componentWillUnmount () {
    super.componentWillUnmount()
    document.removeEventListener('keydown', this._keyDownListener)
    document.removeEventListener('keyup', this._keyUpListener)
    Session.label3dList.unsubscribe(this._drawableUpdateCallback)
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    const { classes } = this.props

    let canvas = (<canvas
      key={`label3d-canvas-${this.props.id}`}
      className={classes.label3d_canvas}
      ref={(ref) => { this.initializeRefs(ref) }}
      onMouseDown={(e) => { this.onMouseDown(e) }}
      onMouseUp={(e) => { this.onMouseUp(e) }}
      onMouseMove={(e) => { this.onMouseMove(e) }}
      onDoubleClick={(e) => { this.onDoubleClick(e) }}
    />)

    if (this.display) {
      const displayRect = this.display.getBoundingClientRect()
      canvas = React.cloneElement(
        canvas,
        { height: displayRect.height, width: displayRect.width }
      )
    }

    return canvas
  }

  /**
   * Handles canvas redraw
   * @return {boolean}
   */
  public redraw (): boolean {
    if (this.canvas) {
      const sensor =
        this.state.user.viewerConfigs[this.props.id].sensor
      if (isCurrentFrameLoaded(this.state, sensor)) {
        this.updateRenderer()
        this.renderThree()
      } else if (this.renderer) {
        this.renderer.clear()
      }
    }
    return true
  }

  /**
   * Handle mouse down
   * @param {React.MouseEvent<HTMLCanvasElement>} e
   */
  public onMouseDown (e: React.MouseEvent<HTMLCanvasElement>) {
    if (!this.canvas || this.checkFreeze()) {
      return
    }
    const normalized = normalizeCoordinatesToCanvas(
      e.clientX, e.clientY, this.canvas
    )
    const NDC = convertMouseToNDC(
      normalized[0],
      normalized[1],
      this.canvas
    )
    const x = NDC[0]
    const y = NDC[1]
    if (this._labelHandler.onMouseDown(x, y)) {
      e.stopPropagation()
    }
  }

  /**
   * Handle mouse up
   * @param {React.MouseEvent<HTMLCanvasElement>} e
   */
  public onMouseUp (e: React.MouseEvent<HTMLCanvasElement>) {
    if (!this.canvas || this.checkFreeze()) {
      return
    }
    if (this._labelHandler.onMouseUp()) {
      e.stopPropagation()
    }
  }

  /**
   * Handle mouse move
   * @param {React.MouseEvent<HTMLCanvasElement>} e
   */
  public onMouseMove (e: React.MouseEvent<HTMLCanvasElement>) {
    if (!this.canvas || this.checkFreeze()) {
      return
    }

    const normalized = normalizeCoordinatesToCanvas(
      e.clientX, e.clientY, this.canvas
    )

    const newX = normalized[0]
    const newY = normalized[1]

    const NDC = convertMouseToNDC(
      newX,
      newY,
      this.canvas
    )
    const x = NDC[0]
    const y = NDC[1]

    this.camera.updateMatrixWorld(true)
    this._raycaster.setFromCamera(new THREE.Vector2(x, y), this.camera)

    const shapes = Session.label3dList.raycastableShapes
    const intersects = this._raycaster.intersectObjects(
      // Need to do this middle conversion because ThreeJS does not specify
      // as readonly, but this should be readonly for all other purposes
      shapes as unknown as THREE.Object3D[], false
    )

    const consumed = (intersects && intersects.length > 0) ?
      this._labelHandler.onMouseMove(x, y, intersects[0]) :
      this._labelHandler.onMouseMove(x, y)
    if (consumed) {
      e.stopPropagation()
    }

    Session.label3dList.onDrawableUpdate()
  }

  /**
   * Handle keyboard events
   * @param {KeyboardEvent} e
   */
  public onKeyDown (e: KeyboardEvent) {
    if (this.checkFreeze() || Session.activeViewerId !== this.props.id) {
      return
    }

    this._keyDownMap[e.key] = true

    if (this._labelHandler.onKeyDown(e)) {
      Session.label3dList.onDrawableUpdate()
    }
  }

  /**
   * Handle keyboard events
   * @param {KeyboardEvent} e
   */
  public onKeyUp (e: KeyboardEvent) {
    if (this.checkFreeze() || Session.activeViewerId !== this.props.id) {
      return
    }

    this._keyDownMap[e.key] = true

    if (this._labelHandler.onKeyUp(e)) {
      Session.label3dList.onDrawableUpdate()
    }
  }

  /**
   * notify state is updated
   */
  protected updateState (state: State): void {
    if (this.display !== this.props.display) {
      this.display = this.props.display
      this.forceUpdate()
    }

    // const item = state.task.items[state.user.select.item]
    // const viewerConfig = this.state.user.viewerConfigs[this.props.id]
    // const sensorId = viewerConfig.sensor
    // for (const key of Object.keys(item.labels)) {
    //   const id = Number(key)
    //   if (item.labels[id].sensors.includes(sensorId)) {
    //     const label = Session.label3dList.get(id)
    //     if (label) {
    //       for (const shape of label.shapes()) {
    //         shape.layers.enable(this.props.id)
    //       }
    //     }
    //   }
    // }
    this._labelHandler.updateState(state, state.user.select.item, this.props.id)
  }

  /**
   * Render ThreeJS Scene
   */
  private renderThree () {
    const state = this.state
    const sensor =
      this.state.user.viewerConfigs[this.props.id].sensor
    if (this.renderer && isCurrentFrameLoaded(state, sensor)) {
      this.renderer.render(Session.label3dList.scene, this.camera)
    } else if (this.renderer) {
      this.renderer.clear()
    }
  }

  /**
   * Handle double click
   * @param _e
   */
  private onDoubleClick (e: React.MouseEvent<HTMLCanvasElement>) {
    if (this._labelHandler.onDoubleClick()) {
      e.stopPropagation()
    }
  }

  /**
   * Set references to div elements and try to initialize renderer
   * @param {HTMLDivElement} component
   * @param {string} componentType
   */
  private initializeRefs (component: HTMLCanvasElement | null) {
    const viewerConfig = this.state.user.viewerConfigs[this.props.id]
    const sensor = viewerConfig.sensor
    if (!component || !isCurrentFrameLoaded(this.state, sensor)) {
      return
    }

    if (viewerConfig.type === ViewerConfigTypeName.IMAGE_3D ||
        viewerConfig.type === ViewerConfigTypeName.HOMOGRAPHY) {
      this.data2d = true
    } else {
      this.data2d = false
    }

    if (component.nodeName === 'CANVAS') {
      if (this.canvas !== component) {
        this.canvas = component
        const rendererParams = {
          canvas: this.canvas,
          alpha: true,
          antialias: true
        }
        this.renderer = new THREE.WebGLRenderer(rendererParams)
        this.forceUpdate()
      }

      if (this.canvas && this.display && this.data2d) {
        const img3dConfig = viewerConfig as Image3DViewerConfigType
        if (img3dConfig.viewScale >= MIN_SCALE &&
            img3dConfig.viewScale < MAX_SCALE) {
          const newParams =
            updateCanvasScale(
              this.state,
              this.display,
              this.canvas,
              null,
              img3dConfig,
              img3dConfig.viewScale / this.scale,
              false
            )
          this.scale = newParams[3]
        }
      } else if (this.display) {
        this.canvas.removeAttribute('style')
        const displayRect = this.display.getBoundingClientRect()
        this.canvas.width = displayRect.width
        this.canvas.height = displayRect.height
      }

      this.updateRenderer()
    }
  }

  /**
   * Update rendering constants
   */
  private updateRenderer () {
    if (this.canvas && this.renderer) {
      this.renderer.setSize(
        this.canvas.width,
        this.canvas.height
      )
    }
  }
}

export default withStyles(styles, { withTheme: true })(Label3dCanvas)
