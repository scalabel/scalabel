// tslint:disable:no-any
// TODO: remove the disable tag
import CssBaseline from '@material-ui/core/CssBaseline'
import { withStyles } from '@material-ui/core/styles/index'
import * as React from 'react'
import SplitPane from 'react-split-pane'
import { setMenuPosition } from '../action/common'
import Session from '../common/session'
import { LayoutStyles } from '../styles/label'
import { ContextMenu } from './context_menu'
import LabelPane from './label_pane'
import PlayerControl from './player_control'

interface ClassType {
  /** title bar */
  titleBar: string
  /** everything below title bar */
  main: string
  /** interface container */
  interfaceContainer: string
  /** pane container */
  paneContainer: string
}

interface Props {
  /** The title bar */
  titleBar: any
  /** The top part of the left side bar */
  leftSidebar1: any
  /** The bottom part of the left side bar */
  leftSidebar2?: any
  /** The bottom bar */
  bottomBar?: any
  /** The top part of the right side bar */
  rightSidebar1?: any
  /** The bottom part of the right side bar */
  rightSidebar2?: any
  /** class type */
  classes: ClassType
}

interface State {
  /** The width of the left side bar */
  left_size: number
  /** The height of the center side bar */
  center_size: number
  /** The width of the right side bar */
  right_size: number
}

interface LayoutState {
  /** The width of the left side bar */
  left_size: number
  /** The height of the center side bar */
  center_size: number
  /** The width of the right side bar */
  right_size: number
}

(window as any).__MUI_USE_NEXT_TYPOGRAPHY_VARIANTS__ = true

/**
 * Layout of the labeling interface
 */
class LabelLayout extends React.Component<Props, State> {
  /** The state of the layout */
  public layoutState: LayoutState

  /**
   * @param {object} props
   */
  constructor (props: any) {
    super(props)
    this.layoutState = { left_size: 0, center_size: 0, right_size: 0 }
    Session.subscribe(this.onStateUpdated.bind(this))
    document.onkeydown = this.disableKeyEvents
  }

  /**
   * called on redux store update
   */
  public onStateUpdated () {
    this.setState(this.layoutState)
  }

  /**
   * Handler on change
   * @param {number} size
   * @param {string} position
   */
  public handleOnChange (size: number, position: string) {
    const layoutState = this.layoutState
    if (position === 'left' && this.layoutState.left_size !== size) {
      layoutState.left_size = size
    } else if (position === 'center' && this.layoutState.center_size !== size) {
      layoutState.center_size = size
    } else if (position === 'right' && this.layoutState.right_size !== size) {
      layoutState.right_size = size
    }
    this.setState(layoutState)
  }

  /**
   * Split component with the second component optional
   * @param {string} split - horizontal or vertical
   * @param {React.Fragment} comp1 - the first component
   * @param {React.Fragment} comp2 - the second component
   * @param {string} name1 - the class name of the first component
   * @param {string} name2 - the class name of the second component
   * @param {number} min - the minimum size
   * @param {number} dflt - the default size
   * @param {number} max - the maximum size
   * @param {string} primary - which component the size constraint is for
   * the second component
   * @param {string} position - left, center or right:
   * which size to update in layoutState
   * @return {Component}
   */
  public optionalSplit (split: 'vertical' | 'horizontal',
                        comp1: React.ReactFragment | undefined,
                        comp2: React.ReactFragment | undefined,
                        name1: string, name2: string, min: number, dflt: number,
                        max: number, primary: 'first' | 'second' = 'first',
                        position: string = 'center') {
    if (!comp1) {
      return
    }
    return (
        comp2 ?
            <SplitPane split={split} minSize={min}
                       defaultSize={dflt}
                       maxSize={max} primary={primary}
                       onChange={(size) => {
                         this.handleOnChange(size, position)
                       }}>
              <div className={name1}>
                {comp1}
              </div>
              <div className={name2}>
                {comp2}
              </div>
            </SplitPane>
            : <div className={name1}>
              {comp1}
            </div>
    )
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    const {
      titleBar,
      leftSidebar1,
      leftSidebar2,
      bottomBar,
      rightSidebar1,
      rightSidebar2,
      classes
    } = this.props
    const leftDefaultWidth = 200
    const leftMaxWidth = 300
    const leftMinWidth = 180
    const rightDefaultWidth = 200
    const rightMaxWidth = 300
    const rightMinWidth = 180
    const topDefaultHeight = 200
    const topMaxHeight = 300
    const topMinHeight = 180
    const bottomDefaultHeight = 200
    const bottomMaxHeight = 300
    const bottomMinHeight = 180

    const playerControl = (<PlayerControl key='player-control'
      num_frames={Session.getState().task.items.length}
    />)

    const state = Session.getState()
    const menuPosition = state.user.layout.menuPosition
    const showMenu = menuPosition.x >= 0 && menuPosition.y >= 0

    const labelInterface = (
      <div className={this.props.classes.interfaceContainer}>
        <div
          className={this.props.classes.paneContainer}
          onMouseDown={(e) => {
            if (e.button === 2) {
              const rect = e.currentTarget.getBoundingClientRect()
              Session.dispatch(setMenuPosition({
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
              }))
            } else if (showMenu) {
              Session.dispatch(setMenuPosition(
                { x: -1, y: -1 }
              ))
            }
          }}
        >
          <LabelPane
            pane={state.user.layout.rootPane} key={'rootPane'}
          />
          <div
            style={{
              position: 'relative',
              top: `${menuPosition.y}px`,
              left: `${menuPosition.x}px`,
              visibility: (showMenu) ? 'visible' : 'hidden',
              background: 'white',
              width: '200px'
            }}
            onMouseDown={(e) => {
              e.stopPropagation()
            }}
          >
            <ContextMenu />
          </div>
        </div>
        { playerControl }
      </div >
    )

    return (
        <React.Fragment>
          <CssBaseline />
          <div className={classes.titleBar}>
            {titleBar}
          </div>
          <main className={classes.main}>
            {
              this.optionalSplit('vertical',
                // left sidebar
                this.optionalSplit('horizontal',
                  leftSidebar1,
                  leftSidebar2,
                  'leftSidebar1',
                  'leftSidebar2',
                  topMinHeight,
                  topDefaultHeight,
                  topMaxHeight,
                  'first'
                ),

                this.optionalSplit('vertical',
                  // center
                  this.optionalSplit('horizontal',
                    labelInterface,
                    bottomBar,
                    'main',
                    'bottomBar',
                    bottomMinHeight,
                    bottomDefaultHeight,
                    bottomMaxHeight,
                    'second',
                    'center'
                  ),

                  // right sidebar
                  this.optionalSplit('horizontal',
                    rightSidebar1,
                    rightSidebar2,
                    'rightSidebar1',
                    'rightSidebar2',
                    topMinHeight,
                    topDefaultHeight,
                    topMaxHeight
                  ),

                  'center',
                  'rightSidebar',
                  rightMinWidth,
                  rightDefaultWidth,
                  rightMaxWidth,
                  'second',
                  'right'
                ),

                'leftSidebar',
                'centerAndRightSidebar',
                leftMinWidth,
                leftDefaultWidth,
                leftMaxWidth,
                'first',
                'left'
              )
            }
          </main>
          {/* End footer */}
        </React.Fragment>
    )
  }

  /** Add listener to window to stop keyboard listeners */
  private disableKeyEvents (e: KeyboardEvent) {
    const state = Session.getState()
    const menuPosition = state.user.layout.menuPosition
    const showMenu = menuPosition.x >= 0 && menuPosition.y >= 0
    if (showMenu) {
      e.stopImmediatePropagation()
    }
  }
}

// LabelLayout.propTypes = {
//   titleBar: PropTypes.object.isRequired,
//   leftSidebar1: PropTypes.object.isRequired,
//   leftSidebar2: PropTypes.object,
//   main: PropTypes.object.isRequired,
//   bottomBar: PropTypes.object,
//   rightSidebar1: PropTypes.object,
//   rightSidebar2: PropTypes.object,
// };

export default withStyles(
  LayoutStyles, { withTheme: true })(LabelLayout)
