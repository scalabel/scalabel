import createStyles from '@material-ui/core/styles/createStyles'
import { withStyles } from '@material-ui/core/styles/index'
import * as React from 'react'
import * as THREE from 'three'
import { Label3DList } from '../drawable/label3d_list'
import { getCurrentPointCloudViewerConfig, isItemLoaded } from '../functional/state_util'
import { PointCloudViewerConfigType, State } from '../functional/types'
import { convertMouseToNDC, updateThreeCameraAndRenderer } from '../view/point_cloud'
import { Viewer } from './viewer'

const styles = () => createStyles({
  canvas: {
    position: 'absolute',
    height: '100%',
    width: '100%'
  }
})

interface ClassType {
  /** CSS canvas name */
  canvas: string
}

interface Props {
  /** CSS class */
  classes: ClassType
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
class Label3dViewer extends Viewer<Props> {
  /** Canvas to draw on */
  private canvas: HTMLCanvasElement | null
  /** ThreeJS Renderer */
  private renderer?: THREE.WebGLRenderer
  /** ThreeJS Scene object */
  private scene: THREE.Scene
  /** ThreeJS Camera */
  private camera: THREE.PerspectiveCamera
  /** ThreeJS sphere mesh for indicating camera target location */
  private target: THREE.Mesh
  /** ThreeJS raycaster */
  private raycaster: THREE.Raycaster
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }

  /** drawable label list */
  private _labels: Label3DList

  /** Ref Handler */
  private refInitializer:
    (component: HTMLCanvasElement | null) => void

  /** UI onr */
  private mouseDownHandler: (e: React.MouseEvent<HTMLCanvasElement>) => void
  /** UI onr */
  private mouseUpHandler: (e: React.MouseEvent<HTMLCanvasElement>) => void
  /** UI onr */
  private mouseMoveHandler: (e: React.MouseEvent<HTMLCanvasElement>) => void
  /** UI onr */
  private keyDownHandler: (e: KeyboardEvent) => void
  /** UI onr */
  private keyUpHandler: (e: KeyboardEvent) => void

  /**
   * Constructor, ons subscription to store
   * @param {Object} props: react props
   */
  constructor (props: Readonly<Props>) {
    super(props)
    this.scene = new THREE.Scene()
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
    this.target = new THREE.Mesh(
      new THREE.SphereGeometry(0.03),
        new THREE.MeshBasicMaterial({
          color:
            0xffffff
        }))
    this.scene.add(this.target)

    this.raycaster = new THREE.Raycaster()
    this.raycaster.near = 1.0
    this.raycaster.far = 100.0

    this.canvas = null

    this._keyDownMap = {}

    this._labels = new Label3DList()

    this.refInitializer = this.initializeRefs.bind(this)

    this.mouseDownHandler = this.onMouseDown.bind(this)
    this.mouseUpHandler = this.onMouseUp.bind(this)
    this.mouseMoveHandler = this.onMouseMove.bind(this)
    this.keyDownHandler = this.onKeyDown.bind(this)
    this.keyUpHandler = this.onKeyUp.bind(this)
  }

  /**
   * Mount callback
   */
  public componentDidMount () {
    document.onkeydown = this.keyDownHandler
    document.onkeyup = this.keyUpHandler
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    const { classes } = this.props
    return (
      <canvas className={classes.canvas} ref={this.refInitializer}
        onMouseDown={this.mouseDownHandler} onMouseUp={this.mouseUpHandler}
        onMouseMove={this.mouseMoveHandler}
      />
    )
  }

  /**
   * Handles canvas redraw
   * @return {boolean}
   */
  public redraw (): boolean {
    const state = this.state.session
    const item = state.user.select.item
    const loaded = state.session.items[item].loaded
    if (loaded) {
      if (this.canvas) {
        this.updateRenderer()
        this.renderThree()
      }
    }
    return true
  }

  /**
   * notify state is updated
   */
  protected updateState (state: State): void {
    this._labels.updateState(state, state.user.select.item)
  }

  /**
   * Render ThreeJS Scene
   */
  private renderThree () {
    if (this.renderer) {
      this.scene.children = []
      this._labels.render(this.scene)
      this.renderer.render(this.scene, this.camera)
    }
  }

  /**
   * Handle mouse down
   * @param {React.MouseEvent<HTMLCanvasElement>} e
   */
  private onMouseDown (e: React.MouseEvent<HTMLCanvasElement>) {
    if (this._labels.onMouseDown()) {
      e.stopPropagation()
    }
  }

  /**
   * Handle mouse up
   * @param {React.MouseEvent<HTMLCanvasElement>} e
   */
  private onMouseUp (e: React.MouseEvent<HTMLCanvasElement>) {
    if (this._labels.onMouseUp()) {
      e.stopPropagation()
    }
  }

  /**
   * Handle mouse move
   * @param {React.MouseEvent<HTMLCanvasElement>} e
   */
  private onMouseMove (e: React.MouseEvent<HTMLCanvasElement>) {
    if (!this.canvas) {
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

    if (this._labels.onMouseMove(x, y, this.camera, this.raycaster)) {
      e.stopPropagation()
    }

    this.renderThree()
  }

  /**
   * Handle keyboard events
   * @param {KeyboardEvent} e
   */
  private onKeyDown (e: KeyboardEvent) {
    this._keyDownMap[e.key] = true

    if (this._labels.onKeyDown(e)) {
      this.renderThree()
    }
  }

  /**
   * Handle keyboard events
   * @param {KeyboardEvent} e
   */
  private onKeyUp (e: KeyboardEvent) {
    this._keyDownMap[e.key] = true

    if (this._labels.onKeyUp(e)) {
      this.renderThree()
    }
  }

  /**
   * Set references to div elements and try to initialize renderer
   * @param {HTMLDivElement} component
   * @param {string} componentType
   */
  private initializeRefs (component: HTMLCanvasElement | null) {
    if (!component) {
      return
    }

    if (component.nodeName === 'CANVAS') {
      this.canvas = component
    }

    if (this.canvas) {
      const rendererParams = { canvas: this.canvas, alpha: true }
      this.renderer = new THREE.WebGLRenderer(rendererParams)
      if (isItemLoaded(this.state.session)) {
        this.updateRenderer()
      }
    }
  }

  /**
   * Update rendering constants
   */
  private updateRenderer () {
    if (this.canvas && this.renderer) {
      const config: PointCloudViewerConfigType = this.getCurrentViewerConfig()
      updateThreeCameraAndRenderer(
        this.canvas,
        config,
        this.renderer,
        this.camera,
        this.target
      )
    }
  }

  /**
   * Get point cloud view config
   */
  private getCurrentViewerConfig (): PointCloudViewerConfigType {
    return (getCurrentPointCloudViewerConfig(this.state.session))
  }
}

export default withStyles(styles, { withTheme: true })(Label3dViewer)
