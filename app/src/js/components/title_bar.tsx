import * as fa from '@fortawesome/free-solid-svg-icons/index'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { AppBar, IconButton, Toolbar, Tooltip } from '@material-ui/core'
import Fade from '@material-ui/core/Fade'
import { Theme } from '@material-ui/core/styles'
import createStyles from '@material-ui/core/styles/createStyles'
import { withStyles } from '@material-ui/core/styles/index'
import Typography from '@material-ui/core/Typography'
import _ from 'lodash'
import React from 'react'
import { connect } from 'react-redux'
import { StateWithHistory } from 'redux-undo'
import { submit } from '../action/common'
import * as selector from '../common/selector'
import Session from '../common/session'
import { Synchronizer } from '../common/synchronizer'
import { Key } from '../common/types'
import { State } from '../functional/types'
import { defaultAppBar } from '../styles/general'
import { StatusMessageBox } from '../styles/label'
import { Component } from './component'

// how long to wait until saving times out
export const saveTimeout = 20000

interface ClassType {
  /** App bar class */
  appBar: string,
  /** Grow class for spacing */
  grow: string,
  /** Title unit class */
  titleUnit: string
}

interface StyleProps {
  /** Styles of TitleBar */
  classes: ClassType
  /** Theme of TitleBar */
  theme: Theme
}

interface StateProps {
  /** Title of TitleBar */
  title: string
  /** DashboardLink of TitleBar */
  dashboardLink: string
  /** InstructionLink of TitleBar */
  instructionLink: string
  /** Whether to show save button or to autosave */
  autosave: boolean
  /** Text for the status banner */
  statusText: string
  /** Whether to hide or show status text */
  statusTextHide: boolean
}

interface DispatchProps {
  /** Function for submitting all progress */
  submit: () => void
}

interface DependencyProps {
  /** Syncrhonizer for saving */
  synchronizer: Synchronizer
}

interface ButtonInfo {
  /** Name */
  title: string,
  /** Icon */
  icon: fa.IconDefinition,
  /** Link */
  href?: string,
  /** Listener  */
  onClick?: () => void
}

/**
 * Convert info of a button to a renderable button
 */
function renderButton (button: ButtonInfo, titleUnit: string): JSX.Element {
  const onClick = _.get(button, 'onClick', undefined)
  const href = _.get(button, 'href', '#')
  const target = ('href' in button ? 'view_window' : '_self')
  return (
    <Tooltip title={button.title} key={button.title}>
      <IconButton className={titleUnit} onClick={onClick}
                  href={href} target={target} data-testid={button.title}>
        <FontAwesomeIcon icon={button.icon} size='xs'/>
      </IconButton>
    </Tooltip>
  )
}

type Props = StyleProps & StateProps & DispatchProps & DependencyProps

/**
 * Title bar
 */
class TitleBar extends Component<Props> {
  /** Listener for key down events */
  private _keyDownListener: (e: KeyboardEvent) => void

  /**
   * Constructor
   * @param {Object} props: react props
   */
  constructor (props: Props) {
    super(props)
    this._keyDownListener = ((e: KeyboardEvent) => {
      if (e.key === Key.S_LOW || e.key === Key.S_UP) {
        this.save()
      }
    })
  }

  /** Mount override */
  public componentDidMount () {
    document.addEventListener('keydown', this._keyDownListener)
  }

  /**
   * Unmount
   * Disables asynchronous callbacks
   */
  public componentWillUnmount () {
    super.componentWillUnmount()
    document.removeEventListener('keydown', this._keyDownListener)
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    const { classes } = this.props
    const { title } = this.props
    const { instructionLink } = this.props
    const { dashboardLink } = this.props
    const { autosave } = this.props
    const { statusText } = this.props
    const { statusTextHide } = this.props

    const buttonInfo: ButtonInfo[] = [
      { title: 'Instructions', href: instructionLink, icon: fa.faInfo },
      { title: 'Keyboard Usage', icon: fa.faQuestion },
      { title: 'Dashboard', href: dashboardLink, icon: fa.faList }
    ]

    const submitHandler = () => {
      this.props.submit()
      // save after submitting, so submit flag is also saved
      if (!autosave) {
        this.save()
      }
    }
    buttonInfo.push(
      { title: 'Submit', onClick: submitHandler, icon: fa.faCheck })

    if (!autosave) {
      buttonInfo.push(
        { title: 'Save', onClick: () => { this.save() }, icon: fa.faSave })
    }

    const buttons = buttonInfo.map((b) => renderButton(b, classes.titleUnit))

    return (
      <AppBar className={classes.appBar}>
        <Toolbar>
          <Typography variant='h6' noWrap>
            {title}
          </Typography>
          <Fade in={!statusTextHide} timeout={300}>
            <StatusMessageBox>
              {statusText}
            </StatusMessageBox>
          </Fade>
          <div className={classes.grow}/>
          {buttons}
        </Toolbar>
      </AppBar>
    )
  }

  /** Save task */
  private save () {
    this.props.synchronizer.sendQueuedActions()
    return
  }
}

const mapStateToProps = (state: StateWithHistory<State>): StateProps => {
  return {
    title: selector.getPageTitle(state),
    instructionLink: selector.getInstructionLink(state),
    dashboardLink: selector.getDashboardLink(state),
    autosave: selector.getAutosaveFlag(state),
    statusText: selector.getStatusText(state),
    statusTextHide: selector.shouldStatusTextHide(state)
  }
}

const mapDispatchToProps = () => {
  return {
    submit: () => Session.dispatch(submit())
  }
}

const styles = (theme: Theme) => createStyles({
  appBar: {
    ...defaultAppBar,
    position: 'relative',
    height: '100%'
  },
  grow: {
    flexGrow: 1
  },
  titleUnit: {
    color: '#bbbbbb',
    margin: theme.spacing(0) * 0.5
  }
})

const styledBar = withStyles(styles, { withTheme: true })(TitleBar)
export default connect(mapStateToProps, mapDispatchToProps)(styledBar)
