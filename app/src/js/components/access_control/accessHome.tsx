import { IconButton, ListItemText, Menu, MenuItem, Toolbar } from '@material-ui/core'
import AppBar from '@material-ui/core/AppBar'
import List from '@material-ui/core/List'
import ListItem from '@material-ui/core/ListItem'
import { withStyles } from '@material-ui/core/styles'
import AccountCircle from '@material-ui/icons/AccountCircle'
import { Match, RouteComponentProps } from '@reach/router'
import React from 'react'
import { accessStyle } from '../../styles/access'

interface ClassType {
  /** body class */
  body: string
  /** sublayout class */
  subLayout: string,
  /** mainContent class */
  mainContent: string
}

interface Props extends RouteComponentProps {
  /** Render class */
  classes: ClassType
}

interface AccessHomeState {
  /** Auth  */
  auth: boolean,
  /** Anchor element for account menu */
  anchorEl: HTMLElement | null,
  /** Current selected menu item */
  selectedIndex: number
}

/**
 * Access home container
 */
class AccessHome extends React.Component<Props, AccessHomeState> {
  constructor (props: Props) {
    super(props)
    this.state = {
      auth: true,
      anchorEl: null,
      selectedIndex: 0
    }
  }

  /**
   * public render method
   */
  public render () {
    // const name = sessionStorage.getItem('username') || ''
    // const authority =
    //   parseInt(sessionStorage.getItem('authority') || '100', 10)
    const { classes } = this.props
    return (
      <div className={classes.body}>
          <AppBar position='static'>
            <Toolbar>
              <IconButton edge='start' color='inherit'>
                <img />
              </IconButton>
              Scalabel User Admin
              {this.state.auth && (
                <div style={{ marginLeft: 'auto' }}>
                  <IconButton
                    aria-label='account of current user'
                    aria-controls='menu-appbar'
                    aria-haspopup='true'
                    onClick={this.handleMenu}
                    color='inherit'
                  >
                    <AccountCircle />
                  </IconButton>
                  <Menu
                    id='menu-appbar'
                    anchorEl={this.state.anchorEl}
                    getContentAnchorEl={null}
                    anchorOrigin={{
                      vertical: 'bottom',
                      horizontal: 'center'
                    }}
                    keepMounted
                    transformOrigin={{
                      vertical: 'top',
                      horizontal: 'center'
                    }}
                    open={Boolean(this.state.anchorEl)}
                    onClose={this.handleClose}
                  >
                    <MenuItem onClick={this.signOut}>Logout</MenuItem>
                  </Menu>
                </div>
              )}
            </Toolbar>
          </AppBar>
          <Match path='/login'>
            {
              (props) =>
              props.match ? (
              <div className={classes.subLayout}>
                  {this.props.children}
              </div>
              ) : (
              <div className={classes.subLayout}>
                <List component='nav'>
                    <ListItem
                      button
                      selected={this.state.selectedIndex === 0}
                      onClick={
                        (event) => { this.handleItemListIndex(event, 0) }
                      }
                    >
                      <ListItemText primary='User management'/>
                    </ListItem>
                    <ListItem
                      button
                      selected={this.state.selectedIndex === 1}
                      onClick={
                        (event) => { this.handleItemListIndex(event, 1) }
                      }
                    >
                      <ListItemText primary='Role management'/>
                    </ListItem>
                    <ListItem
                      button
                      selected={this.state.selectedIndex === 2}
                      onClick={
                        (event) => { this.handleItemListIndex(event, 2) }
                      }
                    >
                      <ListItemText primary='Permission management'/>
                    </ListItem>
                </List>
                <section>
                    <div className={classes.mainContent}>
                        {this.props.children}
                    </div>
                </section>
              </div>
              )
            }
          </Match>
      </div>
    )
  }

  /**
   * open account menu
   * @param {React.MouseEvent<HTMLElement>} event - mouse event
   */
  private handleMenu = (event: React.MouseEvent<HTMLElement>) => {
    this.setState({
      anchorEl: event.currentTarget
    })
  }

  /**
   * close account menu
   */
  private handleClose = () => {
    this.setState({ anchorEl : null })
  }

  /**
   * click item list
   * @param {React.MouseEvent<HTMLDivElement, MouseEvent>} event - mouse event
   * @param {number} index - menu item index
   */
  private handleItemListIndex = (
    _event: React.MouseEvent<HTMLDivElement, MouseEvent>,
    index: number
  ) => {
    this.setState({ selectedIndex: index })
  }

  /**
   * sign out current user
   */
  private signOut = () => {
    fetch('/api/auth/logout', {
      method: 'post',
      headers: { 'Content-Type': 'application/json' }
    })
    .then((response) => response.json())
    .then((data) => {
      if (data.code === 200) {
        // TODO: clear from local storage or redux store
        window.location.href = '/login'
      }
    })
    .catch()
  }
}
export default withStyles(accessStyle, { withTheme: true })(AccessHome)
