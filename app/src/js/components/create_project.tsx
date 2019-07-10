import { Grid, Link } from '@material-ui/core'
import AppBar from '@material-ui/core/AppBar'
import CssBaseline from '@material-ui/core/CssBaseline'
import Drawer from '@material-ui/core/Drawer'
import List from '@material-ui/core/List'
import ListItem from '@material-ui/core/ListItem'
import ListItemText from '@material-ui/core/ListItemText'
import { withStyles } from '@material-ui/core/styles'
import Toolbar from '@material-ui/core/Toolbar'
import Typography from '@material-ui/core/Typography'
import React from 'react'
import { getProjects, toProject } from '../common/service'
import { createStyle, formStyle } from '../styles/create'
import CreateForm from './create_form'

/* Retrieve data from backend */
const projectsToExpress = getProjects()

interface ClassType {
  /** root class */
  root: string

  /** top bar on page */
  appBar: string

  /** sidebar class */
  drawer: string

  /** sidebar background */
  drawerPaper: string

  /** sidebar header */
  drawerHeader: string

  /** class for main content */
  content: string

  /** list header (existing projects) */
  listHeader: string

  /** class used for coloring alternating list item */
  coloredListItem: string

}

interface Props {
  /** Create classes */
  classes: ClassType
}

/**
 * Component which display the create page
 * @param {object} props
 * @return component
 */
function Create (props: Props) {
  const { classes } = props

  /**
   * render function, drawer code from
   * https://material-ui.com/components/drawers/
   * @return component
   */
  return (
    <div className={classes.root}>
      <CssBaseline />
      <AppBar
        position='fixed'
        className={classes.appBar}
      >
        <Toolbar>
          <Typography variant='h6' noWrap>
            Create A Project
                    </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        className={classes.drawer}
        variant='permanent'
        anchor='left'
        classes={{
          paper: classes.drawerPaper
        }}
      >
        <div className={classes.drawerHeader} />
        <List>
          <ListItem>
            <ListItemText primary={'Existing Projects'}
              className={classes.listHeader} />
          </ListItem>
          {projectsToExpress.map((project, index) => (
            <ListItem button
              key={project}
              alignItems='center'
              className={!(index % 2) ?
                classes.coloredListItem : ''}>
              < Grid container justify='center'>
                <Link
                  component='button'
                  variant='body2'
                  color='inherit'
                  onClick={() => {
                    toProject(project)
                  }}
                >
                  {project}
                </Link>
              </Grid>
            </ListItem>
          ))}
        </List>
      </Drawer>

      <main className={classes.content}>
        <StyledForm />
      </main>
    </div>
  )
}

const StyledForm = withStyles(formStyle)(CreateForm)

/** export Dashboard */
export default withStyles(createStyle, { withTheme: true })(Create)
