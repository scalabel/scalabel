import React from 'react';
import PropTypes from 'prop-types';
import {AppBar, CssBaseline, Toolbar, Typography, withStyles, IconButton,
  MenuIcon, Tooltip} from '@material-ui/core';
import SplitPane from 'react-split-pane';
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome';
import {faInfo, faQuestion, faSave, faCheck} from
      '@fortawesome/free-solid-svg-icons';

const styles = (theme) => ({
  appBar: {
    position: 'relative',
    background: '#222222',
  },
  grow: {
    flexGrow: 1,
  },
  btnTitle: {
    color: '#bbbbbb',
    margin: theme.spacing.unit,
  },
});

/**
 * React label interface
 * @param {Object} props
 * @return {*}
 * @constructor
 */
function LabelLayout(props) {
  const {classes} = props;

  return (
      <React.Fragment>
        <CssBaseline />
        <AppBar position="static" className={classes.appBar}>
          <Toolbar variant="dense">
            <IconButton className={classes.menuButton}
                        color="inherit" aria-label="Menu">
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" color="inherit">
              BDD Labeling Session
            </Typography>
            <div className={classes.grow} />
            <Tooltip title='Instructions'>
              <IconButton className={classes.btnTitle}>
                <FontAwesomeIcon icon={faInfo} size='xs'/>
              </IconButton>
            </Tooltip>
            <Tooltip title='Keyboard Usage'>
              <IconButton className={classes.btnTitle}>
                <FontAwesomeIcon icon={faQuestion} size='xs'/>
              </IconButton>
            </Tooltip>
            <Tooltip title='Save'>
              <IconButton className={classes.btnTitle}>
                <FontAwesomeIcon icon={faSave} size='xs'/>
              </IconButton>
            </Tooltip>
            <Tooltip title='Submit'>
              <IconButton className={classes.btnTitle}>
                <FontAwesomeIcon icon={faCheck} size='xs'/>
              </IconButton>
            </Tooltip>
          </Toolbar>
        </AppBar>
        <main>
          <SplitPane split='vertical' minSize={180}
                     defaultSize={200} maxSize={300}>
            <div></div>
            <SplitPane split='vertical' minSize={500}
                       defaultSize='80%' maxSize={-200}>
              <div></div>
              <SplitPane split='horizontal' minSize={180}
                         defaultSize={200} maxSize={-180}>
                <div></div>
                <div></div>
              </SplitPane>
            </SplitPane>
          </SplitPane>
        </main>
        {/* End footer */}
      </React.Fragment>
  );
}

LabelLayout.propTypes = {
  classes: PropTypes.object.isRequired,
  theme: PropTypes.object.isRequired,
};

export default withStyles(styles, {withTheme: true})(LabelLayout);
