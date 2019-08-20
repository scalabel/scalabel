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
import Session, { ConnectionStatus } from '../common/session'
import { defaultAppBar } from '../styles/general'
import { StatusMessageBox } from '../styles/label'
import { Component } from './component'

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

interface ClassType {
  /** app bar class */
  appBar: string,
  /** grow class for spacing */
  grow: string,
  /** title unit class */
  titleUnit: string
}

interface Props {
  /** Styles of TitleBar */
  classes: ClassType
  /** Theme of TitleBar */
  theme: Theme
  /** title of TitleBar */
  title: string
  /** dashboardLink of TitleBar */
  dashboardLink: string
  /** instructionLink of TitleBar */
  instructionLink: string
}

/**
 * Save the current state to the server
 */
function save (callerComponent: TitleBar) {
  Session.status = ConnectionStatus.SAVING
  callerComponent.forceUpdate()
  const state = Session.getState()
  const xhr = new XMLHttpRequest()
  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4) {
      Session.status = ConnectionStatus.SAVED
      callerComponent.forceUpdate()
      setTimeout(() => {
        Session.status = ConnectionStatus.UNSAVED
        callerComponent.forceUpdate()
      }, 5000)
      if (JSON.parse(xhr.response) !== 0) {
        alert('Save failed.')
      }
    }
  }
  xhr.open('POST', './postSaveV2')
  xhr.send(JSON.stringify(state))
}

/**
 * turn assistant view on/off
 */
function toggleAssistantView () {
  // Session.dispatch({type: types.TOGGLE_ASSISTANT_VIEW});
}

/**
 * Title bar
 */
class TitleBar extends Component<Props> {
  /**
   * Constructor
   * @param {Object} props: react props
   */
  constructor (props: Props) {
    super(props)
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

    const buttonInfo = [
      { title: 'Instructions', href: instructionLink, icon: fa.faInfo },
      { title: 'Keyboard Usage', icon: fa.faQuestion },
      { title: 'Dashboard', href: dashboardLink, icon: fa.faList },
      {
        title: 'Assistant View', onClick: toggleAssistantView,
        icon: fa.faColumns
      },
      { title: 'Save', onClick: () => { save(this) }, icon: fa.faSave },
      { title: 'Submit', onClick: () => { save(this) }, icon: fa.faCheck }
    ]
    const buttons = buttonInfo.map((b) => {
      const onClick = _.get(b, 'onClick', undefined)
      const href = _.get(b, 'href', '#')
      const target = ('href' in b ? 'view_window' : '_self')
      return (
              <Tooltip title={b.title} key={b.title}>
                <IconButton className={classes.titleUnit} onClick={onClick}
                            href={href} target={target} data-testid={b.title}>
                  <FontAwesomeIcon icon={b.icon} size='xs'/>
                </IconButton>
              </Tooltip>
      )
    })

    let sessionStatus: string
    let hideMessage = false
    switch (Session.status) {
      case ConnectionStatus.SAVING: {
        sessionStatus = 'Saving in progress...'
        break
      }
      case ConnectionStatus.SAVED: {
        sessionStatus = 'All progress saved.'
        break
      }
      case ConnectionStatus.RECONNECTING: {
        sessionStatus = 'Trying to reconnect...'
        break
      }
      case ConnectionStatus.UNSAVED: {
        // Want the SAVED status text during fade animation
        sessionStatus = 'All progress saved.'
        hideMessage = true
        break
      }
      default: {
        sessionStatus = 'Error occured, try refreshing.'
        break
      }
    }

    return (
      <AppBar className={classes.appBar}>
        <Toolbar>
          <Typography variant='h6' noWrap>
            {title}
          </Typography>
          <Fade in={!hideMessage} timeout={300}>
            <StatusMessageBox>
              {sessionStatus}
            </StatusMessageBox>
          </Fade>
          <div className={classes.grow}/>
          {buttons}
        </Toolbar>
      </AppBar>
    )
  }
}

export default withStyles(styles, { withTheme: true })(TitleBar)
