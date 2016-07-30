"""
Define graphical placeholders for table elements.
The main top-level class is Table which can be built from a DataTree 
(UB-tree built from folders of images, see cardo.tree.dTree*). 
It allows insertion of spacers between rows and columns and automatic 
tuning of cell sizes.

Note that the reference size on which all elements of the table are scaled is
the text height which is fixed to BoxedText.DEFAULT_FONT_H.

Basic class is BoxedElement which is a container to handle padding around
an object.
Subclasses of BoxedElement:
- BoxedText: text can be aligned within a virtual bounding box
- BoxedImage: image can be positioned within a virtual bounding box
- Spacer: graphical element whose size should not change

"""
import os.path as op
import imghdr
import struct
import math
from pprint import pformat
import logging
import numpy as np
from cardo import tree

import svgwrite

logger = logging.getLogger('cardo')

class ImageFileNotFound(Exception):
    pass

class BoxedElement(object):

    def __init__(self, box_x, box_y, min_width, min_height):

        #Coords of the top left corner of the box
        self.box_x = box_x
        self.box_y = box_y

        self.box_min_width = min_width
        self.box_min_height = min_height

        self.box_width = min_width
        self.box_height = min_height

    def set_box_x(self, x):
        self.box_x = x

    def set_box_y(self, y):
        self.box_y = y

    def set_box_width(self, bw):
        if bw > self.box_min_width: #Ensure box has at least minimal width
            self.box_width = bw

    def set_box_height(self, bh):
        if bh > self.box_min_height: #Ensure box has at least minimal height
            self.box_height = bh

    def get_box_width(self):
        return self.box_width

    def get_box_height(self):
        return self.box_height   

    def get_box_size(self):
        return self.get_box_width(), self.get_box_height()

    def get_box_coords(self):
        """ Return coordinates of top-left corner """
        return self.box_x, self.box_y
   
    def get_box_right_x(self):
        return self.box_x + self.get_box_width()

    def get_box_bot_y(self):
        return self.box_y + self.get_box_height()

    def get_box_bot_left_coords(self):
        return self.box_x, self.box_y + self.get_box_height()

    @staticmethod
    def hdeoverlap(belems, gap=0):
        """"
        Remove horizontal overlaps between given boxes.
        The 1st box is not moved and all subsequent boxes are moved
        relatively to the previous, from left to right.
        """
        for i, belem in enumerate(belems[1:]):
            belem.set_box_x(belems[i].get_box_right_x() + gap)


    @staticmethod
    def vdeoverlap(belems, gap=0):
        """"
        Remove vertical overlaps between given boxes.
        The 1st box is not moved and all subsequent boxes are moved
        relatively to the previous, from top to bottom.
        """
        for i, belem in enumerate(belems[1:]):
            belem.set_box_y(belems[i].get_box_bot_y() + gap)
           
class BoxedText(BoxedElement):

    DEFAULT_FONT_H = 40 #px
    DEFAULT_FONT = 'Bitstream Vera Sans Mono'
    DEFAULT_FONT_W = 25 #px

    LEFT_ALIGN = 0
    RIGHT_ALIGN = 1
    CENTER_ALIGN = 2   

    def __init__(self, text, box_x=0, box_y=0,
                 box_width=None, box_height=None,
                 font_size_px=DEFAULT_FONT_H,
                 text_halign=CENTER_ALIGN):
        self.text = text
        self.text_halignment = text_halign
        self.font_size_px = font_size_px
       
        self.text_width, self.text_height = self.get_text_size()

        BoxedElement.__init__(self, box_x, box_y, self.text_width,
                              self.text_height)

        if box_width is not None:
            self.set_box_width(box_width)
           
        if box_height is not None:
            self.set_box_height(box_height)       

    def __repr__(self):
        return self.text + str(self.get_box_size())

    def get_text_size(self):
        """ Predict the size of the text bounding box in px"""
        return (len(self.text) * BoxedText.DEFAULT_FONT_W,
                BoxedText.DEFAULT_FONT_H)

    def get_text_coords(self):
        """
        Return the coords of the bottom left corner of the text, which
        depend on its alignement within the surrounding box.
        """
        ytext = self.box_y + self.box_height/2. - self.text_height/2.
        if self.text_halignment == BoxedText.LEFT_ALIGN:
            return (self.box_x, ytext)
        elif self.text_halignment == BoxedText.RIGHT_ALIGN:
            return (self.box_x + self.box_width - self.text_width, ytext)
        elif self.text_halignment == BoxedText.CENTER_ALIGN:
            return (self.box_x + self.box_width/2. - self.text_width/2.,
                    ytext)
        else:
            raise Exception('Uknown text halignment:' + \
                            str(self.text_halignment))

    def to_svg(self):
        style = 'font-style:normal;font-family:%s;font-size:%dpx' \
                %(BoxedText.DEFAULT_FONT, self.font_size_px)
        return svgwrite.text.Text(self.text, insert=self.get_text_coords(),
                                  style=style)


class BoxedRect(BoxedElement):

    HALIGN_CENTER = 0
    HALIGN_LEFT = 1
    HALIGN_RIGHT = 2

    VALIGN_CENTER = 0
    VALIGN_LEFT = 1
    VALIGN_RIGHT = 2
   
    def __init__(self, rwidth, rheight, box_x=0, box_y=0,
                 halign=HALIGN_CENTER, valign=VALIGN_CENTER):

        self.halign = halign
        self.valign = valign

        self.rect_width = rwidth
        self.rect_height = rheight

        BoxedElement.__init__(self, box_x, box_y, self.rect_width,
                              self.rect_height)
       

    def get_rect_coords(self):
       
        if self.halign == BoxedRect.HALIGN_CENTER:
            x = self.box_x +  self.get_box_width()/2. - self.rect_width/2.
        elif self.halign == BoxedRect.HALIGN_LEFT:
            x = self.box_x
        elif self.halign == BoxedRect.HALIGN_RIGHT:
            x = self.box_x +  self.get_box_width() - self.rect_width
        else:
            raise Exception()

        if self.valign == BoxedRect.VALIGN_CENTER:
            y = self.box_y +  self.get_box_height()/2. - self.rect_height/2.
        elif self.valign == BoxedRect.VALIGN_LEFT:
            y = self.box_y
        elif self.valign == BoxedRect.VALIGN_RIGHT:
            y = self.box_y +  self.get_box_height() - self.rect_height
        else:
            raise Exception()
                       
        return (x,y)


class Spacer(BoxedRect):
    #TODO: test
    def __init__(self, width=0, height=0):
        BoxedRect.__init__(self, width, height, 0, 0)

    def __repr__(self):
        return '@' + str(self.get_box_size())
       
    def to_svg(self):
        return svgwrite.shapes.Rect(insert=self.get_box_coords(),
                                    size=self.get_box_size())
       
class BoxedImage(BoxedRect):

    def __init__(self, img_fn, box_x=0, box_y=0, img_w=None, img_h=None,
                 halign=BoxedRect.HALIGN_CENTER, valign=BoxedRect.VALIGN_CENTER):

        if not op.exists(img_fn):
            raise ImageFileNotFound()
        
        self.img_fn = img_fn

        img_orig_w, img_orig_h = get_image_size(img_fn)
        if img_h is None and img_w is None:
            self.rect_width, self.rect_height = img_orig_w, img_orig_h
        elif img_w is not None:
            self.rect_width = img_w
            #maintain aspect ratio
            self.rect_height = round(img_orig_h * 1. / img_orig_w * img_w)
        else: #img_h is not None
            self.rect_height = img_h
            #maintain aspect ratio
            self.rect_width = round(img_orig_w * 1. / img_orig_h * img_h)

        BoxedRect.__init__(self, self.rect_width, self.rect_height,
                           box_x, box_y, halign, valign)

    def __repr__(self):
        return op.basename(self.img_fn)
        
    def to_svg(self):
        return svgwrite.image.Image(self.img_fn, insert=self.get_rect_coords(),
                                    size=(self.rect_width, self.rect_height))
   
def get_image_size(fname):
    """
    Determine the image type of given file name and return its size as
    a tuple of float (width, height).
    """
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                raise Exception('Check "0x0d0a1a0a" dit not pass reading %s' \
                                %fname)
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            fhandle.seek(0) # Read 0xff next
            size = 2
            ftype = 0
            while not 0xc0 <= ftype <= 0xcf:
                fhandle.seek(size, 1)
                byte = fhandle.read(1)
                while ord(byte) == 0xff:
                    byte = fhandle.read(1)
                ftype = ord(byte)
                size = struct.unpack('>H', fhandle.read(2))[0] - 2
            # We are at a SOFn block
            fhandle.seek(1, 1)  # Skip `precision' byte.
            height, width = struct.unpack('>HH', fhandle.read(4))
        else:
            raise Exception('Image type not supported %s:' \
                            %op.splitext(fname)[2])

        return width, height


def adjust_table_sizes(row_hdr_btexts, col_hdr_btexts, bimgs):
   
    # adjust sizes from headers:
    adjust_hdr(row_hdr_btexts + [list(bimgs[:, 0])],
               use_height_for_last_line=True)
    adjust_hdr(col_hdr_btexts + [list(bimgs[0, :])])

    # Adjust spacers
    for icol,e in enumerate(bimgs[0, :]):
        if isinstance(e, Spacer):
            e.set_box_height(bimgs[0, icol-1].get_box_height())
    for irow,e in enumerate(bimgs[:, 0]):
        if isinstance(e, Spacer):
            e.set_box_width(bimgs[irow-1, 0].get_box_width())
    # Set adjusted sizes for all images:
    for irow, row_bimgs in enumerate(bimgs):
        for icol, bimg in enumerate(row_bimgs):
            if icol > 0: # 1st column has been adjusted
                # set height
                bimg.set_box_height(bimgs[irow, 0].get_box_height())
            if irow > 0: # 1st row has been adjusted
                # set width
                bimg.set_box_width(bimgs[0, icol].get_box_width())
    return row_hdr_btexts, col_hdr_btexts, bimgs
   

   


def arrange_table(row_hdr_btexts, col_hdr_btexts, bimgs_array):
    for hdr_btexts in (col_hdr_btexts, row_hdr_btexts):
        # Place header elements next to each other, row by row
        for hdr_line in hdr_btexts:
            BoxedElement.hdeoverlap(hdr_line)
        # Place header elements on top of each others:
        ref_elements = [hdr_line[0] for hdr_line in hdr_btexts]
        for ilevel, hdr_line in enumerate(hdr_btexts):
            if ilevel > 0:
                for btext in hdr_line:
                    BoxedElement.vdeoverlap([ref_elements[ilevel-1],
                                             btext])       
    # Arrange images, by row then by column
    for row_bimgs in bimgs_array:
        BoxedElement.hdeoverlap(row_bimgs)
    for col_bimgs in bimgs_array.T:
        BoxedElement.vdeoverlap([col_hdr_btexts[-1][0]] + list(col_bimgs))

def adjust_hdr(hdr, use_height_for_last_line=False):
    def _adjust_cell(l, i):
        logger.debug('_adjust_cell(%d, %d)', l, i)
        if isinstance(hdr[l][i], Spacer):
            logger.debug('-> Spacer => returning')
            return
        if l == 0:
            level_len = len(hdr[l])
        else:
            level_len = (len(hdr[l]) - (len(hdr[l-1])-1)/2) /   \
                             ((len(hdr[l-1])+1)/2)
        logger.debug('Cell is part of: %s, with unit level len of %d',
                     str(hdr[l]), level_len)

        # compute size based on children       
        children_size = 0
        if l == len(hdr) - 2:  # children are images
            logger.debug('current level %d is parent of images', l)
            # there is a one to one index match from last hdr line
            children_idx = [i]
            if use_height_for_last_line: #special case for row header
                children_size = hdr[l+1][i].get_box_height()
            else:
                children_size = hdr[l+1][i].get_box_width()
        elif l < len(hdr) - 2: # there are children
            logger.debug('current level %d is normal', l)
            clvl_len = (len(hdr[l+1]) - (len(hdr[l])-1)/2) /   \
                               ((len(hdr[l])+1)/2)
            logger.debug('Child level len: %d', clvl_len)
            children_idx = range(i/2*(clvl_len+1), i/2*(clvl_len+1) + clvl_len)
            children_size = sum(hdr[l+1][ic].get_box_width() \
                                 for ic in children_idx)
        else: # l == len(hdr) - 1 # cur level is images (no child)
            logger.debug('current level %d is images (no child)', l)
            children_idx = []
            children_size = 0
           
        logger.debug('Children: %s', str(children_idx))           
        logger.debug('children-based size: %d', children_size)
        # Current size
        if l == len(hdr) - 1 and use_height_for_last_line: # line is images
            current_size = hdr[l][i].get_box_height()
        else:
            current_size = hdr[l][i].get_box_width()
        logger.debug('current cell size: %d', current_size)
       
        # compute size based on parent:
        parent_size = -1
        if l > 0:
            if l == len(hdr) - 1: # line is images
                # one to one index matching
                parent_idx = i
                parent_size = hdr[l-1][parent_idx].get_box_width()
            else:

                logger.debug('Compute total gap size from %d to %d',
                             i-i%(level_len+1), i-i%(level_len+1) + level_len)
                gap_size = sum(hdr[l][k].get_box_width() \
                               for k in xrange(i - i%(level_len+1),
                                               i - i%(level_len+1) + level_len)\
                               if isinstance(hdr[l][k], Spacer))
                logger.debug('Gap size: %d', gap_size)
                parent_idx = ((i - i%(level_len+1)) * 2) / (level_len + 1)
                assert parent_idx < len(hdr[l-1])
                logger.debug('Normal line, get parent size from cell(%d,%d)',
                             l-1, parent_idx)
                parent_size = math.ceil((hdr[l-1][parent_idx].get_box_width() - \
                                         gap_size) / ((level_len+1.)/2))
        else:
            logger.debug('No parent (1st level)')
           
        logger.debug('Parent-based size: %d', parent_size)
           
        new_size = max(children_size, current_size, parent_size)
        logger.debug('# new size of cell(%d,%d) = max(ch=%d, cur=%d, p=%d) = %d',
                     l, i, children_size, current_size, parent_size, new_size)
        level_start = i-i%(level_len+1)
        level_end = i-i%(level_len+1) + level_len
        logger.debug('Compute new level size from (%d,%d) to (%d,%d)',
                     l, level_start, l, level_end)
        if l == len(hdr) - 1 and use_height_for_last_line:
            # line is images and hdr is for rows (adjust height of imgs)
            hdr[l][i].set_box_height(new_size)
            level_size = sum(hdr[l][k].get_box_height() \
                             for k in xrange(level_start, level_end))
        else:
            hdr[l][i].set_box_width(new_size)
            level_size = sum(hdr[l][k].get_box_width() \
                             for k in xrange(level_start, level_end))

        logger.debug('New level size: %d', level_size)
        if parent_size != -1 and \
           level_size > hdr[l-1][parent_idx].get_box_width():
            logger.debug('# new size %d invalidates parent -> '\
                         'Go from (%d,%d) to parent (%d,%d) [%d]',
                         new_size, l, i, l-1, parent_idx,
                         hdr[l-1][parent_idx].get_box_width())
            _adjust_cell(l-1, parent_idx)
        else:
            for ichild in children_idx:
                logger.debug('# Go from (%d,%d) to child (%d,%d)',
                             l, i, l+1, ichild)
                _adjust_cell(l+1, ichild)

    logger.debug('Adjust hdr: %s', pformat(hdr))
    for i in xrange(len(hdr[0])): # Start from 1st line
        _adjust_cell(0, i)
               

class GTree(object):
    """
    Tree of graphic elements which handles balanced resizing 
    and insertion of spacers 
    """

    def __init__(self, gfx_element, children=None, parent=None):
        self.gfx_element = gfx_element

        self.children = children
        if self.children is None:
            self.children = []

        self.parent = parent

    def __repr__(self):
        return self.gfx_element.__repr__()

    def set_parent(self, parent):
        self.parent = parent
        
    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0
        
    def get_height(self):
        if self.is_leaf():
            return 0
        else:
            return max(child.get_height()+1 for child in self.children)

    def to_string(self):
        def _depth_first_walk(node, line):
            if node.is_leaf():
                print line
            else:
                for child in node.children:
                    _depth_first_walk(child, line + '-' + str(node.gfx_element))
        _depth_first_walk(self, '')

        
    @staticmethod
    def from_hdr_and_images(hdr_levels, images):
        """
        Build a GTree from header levels and the grid of images

        Args:
            - hdr_levels (list of list of str):
                header level elements, eg:
                   [['lvl1_item1', 'lvl1_item2, ...],
                    ['lvl2_item1', 'lvl1_item1, ...],
                    ...]
            - images (list of list of BoxedImage):
                 2d array of image elements. Row size must equal
                 the cartesian product of hdr_levels
        """
        def leaf_img_trees():
            for icol in xrange(len(images[0])):
                col_root = GTree(images[0][icol])
                cur_node = col_root
                for irow in xrange(1, len(images)):
                    next_child = GTree(images[irow][icol], parent=cur_node)
                    cur_node.add_child(next_child)
                    cur_node = next_child
                yield col_root
        img_col_tree_iterator = leaf_img_trees()
        
        root = GTree(None)
        def _create_levels_rec(node, levels):
            if len(levels) == 0:
                return
            for element in levels[0]:
                element = BoxedText(element)
                new_child = GTree(element, parent=node)
                node.add_child(new_child)
                _create_levels_rec(new_child, levels[1:])
            if len(levels) == 1:
                for child in node.children:
                    img_tree = img_col_tree_iterator.next()
                    img_tree.set_parent(child)
                    child.add_child(img_tree)
        _create_levels_rec(root, hdr_levels)
        return root

    def add_spacers(self, base_gap, create_spacer, nb_levels_to_space):
                    
        def _make_spacer_stack(node, gap_factor):
            spacer = create_spacer(gap_factor * base_gap,
                                   node.children[0].gfx_element)
            root_spacer_node = GTree(spacer, parent=node)
            cur_spacer_node = root_spacer_node
            cur_node = node.children[0]
            while not cur_node.is_leaf():
                spacer = create_spacer(gap_factor * base_gap,
                                       cur_node.children[0].gfx_element)
                new_node = GTree(spacer, parent=cur_spacer_node)
                cur_spacer_node.add_child(new_node)
                cur_spacer_node = new_node
                cur_node = cur_node.children[0]
            return root_spacer_node
        
        def _space_rec(node, gap_factor):
            logger.debug('_space_rec: height=%d, node: %s', gap_factor, str(node))
            if node.is_leaf():
                return
            new_children = []
            for child in node.children[:-1]:
                _space_rec(child, max(0, gap_factor-1))
                new_children.append(child)                
                spacer_stack = _make_spacer_stack(node, gap_factor)
                spacer_stack.set_parent(node)
                new_children.append(spacer_stack)
            _space_rec(node.children[-1], max(0, gap_factor-1))
            new_children.append(node.children[-1])
            logger.debug('new children of %s: %s', str(node), str(new_children))
            node.children = new_children
            
        _space_rec(self, nb_levels_to_space)

    def adjust_size(self, get_size, set_size):

        if isinstance(self.gfx_element, Spacer):
            logger.debug('Avoid spacer: %s', str(self.gfx_element))
            return

        
        logger.debug('Adjust size of "%s"', str(self))
        logger.debug('Children sizes: "%s"',
                     ",".join(str(get_size(child.gfx_element)) \
                              for child in self.children))
        if self.gfx_element is not None:
            current_size = get_size(self.gfx_element)
            logger.debug('current size: %d', current_size)
            
            child_based_size = sum(get_size(child.gfx_element)
                                   for child in self.children)
            logger.debug('children-based size: %d', child_based_size)
            
            if self.parent is not None and self.parent.gfx_element is not None:
                siblings = self.parent.children
                logger.debug('siblings: %s', str(siblings))
                nb_non_spacer = sum(1 for s in siblings \
                                    if not isinstance(s.gfx_element, Spacer)) * 1.
                logger.debug('nb non-spacers: %d', nb_non_spacer)
                spacer_size = sum(get_size(sib.gfx_element) \
                                  for isib, sib in enumerate(siblings) \
                                  if isinstance(sib.gfx_element, Spacer))
                logger.debug('Gap size: %d', spacer_size)
                par_based_size = math.floor((get_size(self.parent.gfx_element) - \
                                                spacer_size) /                   \
                                               nb_non_spacer)
            else:
                siblings = []
                par_based_size = 0
            logger.debug('parent-based size: %d', par_based_size)
                
            new_size = int(max(current_size, child_based_size, par_based_size))
            set_size(self.gfx_element, new_size)
            logger.debug('new size: %d', new_size)
    
            new_sib_size = sum(get_size(sib.gfx_element) for sib in siblings)
            logger.debug('new siblings size: %d', new_sib_size)
            
        if self.parent is not None and self.parent.gfx_element is not None \
           and new_sib_size > get_size(self.parent.gfx_element):
            self.parent.adjust_size(get_size, set_size)
        else:
            for child in self.children:
                child.adjust_size(get_size, set_size)

    def get_elements(self, filt=None, walk_type='bfs', start_depth=0,
                     max_depth=-1):
        """
        Gather all elements in the tree that match the given predicator "filt".
        Do a breath-first or depth-first walk of the tree.
        """
        if filt is None:
            filt = lambda x: True
            
        if walk_type == 'bfs':
            visited, queue = set(), [(self,-1)]
            elements = []
            cur_level = -1
            while queue:
                node, level = queue.pop(0)
                logger.debug('get_elements visiting: %s', str(node))
                if level != cur_level:
                    if max_depth !=-1 and level >= max_depth:
                        break
                    if level >= start_depth:
                        elements.append([])
                    cur_level = level
                if level >= start_depth and node.gfx_element is not None and \
                   filt(node.gfx_element):
                    elements[-1].append(node.gfx_element)
                if node not in visited:
                    visited.add(node)
                    queue.extend((c, cur_level+1) for c in node.children \
                                 if c not in visited)
        else: # detph-1st-search
            def _depth_first_walk(node, elems, depth):
                if node.is_leaf() or (max_depth != -1 and depth==max_depth):
                    yield elems
                else:
                    for child in node.children:
                        if depth >= start_depth and \
                           node.gfx_element is not None and \
                           filt(node.gfx_element):
                            _depth_first_walk(node, elems + node.gfx_element,
                                              depth + 1)
                        else: #skip tree root
                            _depth_first_walk(node, elems, depth)
            elements = []
            _depth_first_walk(self, elements, 0)
        return elements
                
class Table(object):
    """
    Tree-based representation of a table with row and col headers.

    It allows insertion of gaps of increasing size depending on the position
    in the header and also optimization of cell sizes.

    Internally, it consists of 2 GTree objects: one for rows and one for cols.
    """

    DEFAULT_ROW_BGAP = BoxedText.DEFAULT_FONT_H
    DEFAULT_COL_BGAP = BoxedText.DEFAULT_FONT_W*3

    def __init__(self, row_hdr_levels, col_hdr_levels, images):
        """ 
        Build a table from given headers and grid of images

        ASSUME: images is "aligned" with given headers.
                -> imgage[0, 0] corresponds to:
                      - col_hdr_levels[0][0], col_hdr_levels[1][0],  ...
                        col_hdr_levels[-1][0]
                      and
                      - row_hdr_levels[0][-1], row_hdr_levels[1][-1],  ...
                        row_hdr_levels[-1][-1]
                -> the cartesian product of col_hdr_levels has the same size as
                   the size of the 1st row of images
                -> the cartesian product of row_hdr_levels has the same size as
                   the size of the 1st column of images                
        """
        assert(np.prod([len(lvl) for lvl in col_hdr_levels]) == len(images[0]))
        assert(np.prod([len(lvl) for lvl in row_hdr_levels]) == len(images))
        # Set image size relative to text size:
        # img_h = BoxedText.DEFAULT_FONT_H * 7
        # for row in len(images):
        #     for img in row:
        #         img.set_

        self.col_hdr_height = len(col_hdr_levels)
        self.col_gtree = GTree.from_hdr_and_images(col_hdr_levels, images)
        self.row_hdr_height = len(row_hdr_levels)
        self.row_gtree = GTree.from_hdr_and_images(row_hdr_levels, images.T)

    @staticmethod
    def from_dtree(dtree, root_path, branch_names, row_branches,
                   column_branches):
        return Table(*dtree_to_table_elements(dtree, root_path, branch_names,
                                              row_branches, column_branches))
    
    def add_spacers(self, row_base_gap=DEFAULT_ROW_BGAP,
                    col_base_gap=DEFAULT_COL_BGAP):

        def create_col_spacer(gap, ref_sibling):
            return Spacer(width=gap, height=ref_sibling.get_box_height())
        self.col_gtree.add_spacers(col_base_gap, create_col_spacer,
                                   self.col_hdr_height)
        
        def create_row_spacer(gap, ref_sibling):
            if isinstance(ref_sibling, BoxedImage):
                return Spacer(height=gap, width=ref_sibling.get_box_height())
            else:
                return Spacer(width=gap, height=ref_sibling.get_box_height())    
        self.row_gtree.add_spacers(row_base_gap, create_row_spacer,
                                   self.row_hdr_height)
            
    def adjust_cell_sizes(self):
        """
        Optimize cell sizes of header and images so that all element 
        fit into one another
        """
        def colgt_get_size(gelem):
            return gelem.get_box_width()
        def colgt_set_size(gelem, size):
            gelem.set_box_width(size)
        self.col_gtree.adjust_size(colgt_get_size, colgt_set_size)
            
        def rowgt_get_size(gelem):
            if isinstance(gelem, BoxedImage):
                return gelem.get_box_height()
            else:
                return gelem.get_box_width()
        def rowgt_set_size(gelem, size):
            if isinstance(gelem, BoxedImage):
                gelem.set_box_height(size)
            else:
                gelem.set_box_width(size)
        self.col_gtree.adjust_size(rowgt_get_size, rowgt_set_size)

    def deoverlap(self):
        """
        Move all elements to remove all overlaps.
        For headers, elements are moved from left to right
        For the grid of images, elements are moved from left to right and
        from top to bottom
        """
        elements_by_column = self.col_gtree.get_elements(walk_type='dfs')
        for column in elements_by_column:
            BoxedElement.vdeoverlap()

        row_header = self.row_gtree.get_elements(Table.is_hdr_element,
                                                 walk_type='dfs')
        for row in row_header:
            BoxedElement.hdeoverlap(row)

        imgs_by_row = self.row_gtree.get_elements(Table.is_img_element,
                                                  walk_type='dfs')
        for row in imgs_by_row:
            BoxedElement.hdeoverlap(row)
        
    @staticmethod
    def is_hdr_element(element):
        return not isinstance(element, BoxedImage)

    @staticmethod
    def is_img_element(element):
        return isinstance(element, BoxedImage)
    
    def to_svg(self):

        dwg = svgwrite.Drawing()
        table_group = svgwrite.container.Group(id='table')

        col_hdr_group = svgwrite.container.Group(id='table_col_hdr')
        column_header = self.col_gtree.get_elements(max_depth=self.col_hdr_height)
        for col_hdr_line in column_header:
            for element in col_hdr_line:
                col_hdr_group.add(element.to_svg())            
        table_group.add(col_hdr_group)

        row_hdr_group = svgwrite.container.Group(id='table_row_hdr')
        row_header = self.row_gtree.get_elements(max_depth=self.row_hdr_height)
        for row_hdr_line in row_header:
            for element in row_hdr_line:
                row_hdr_group.add(element.to_svg())

        imgs_cview = self.col_gtree.get_elements(start_depth=self.col_hdr_height)
        imgs_rview = self.row_gtree.get_elements(start_depth=self.row_hdr_height)

        imgs = np.empty((len(row_header[-1]), len(column_header[-1])),
                        dtype=object)
        imgs[::2,:] = np.array(imgs_cview, dtype=object)
        imgs[:,::2] = np.array(imgs_rview, dtype=object).T
        for i,j in np.vstack(np.where(np.equal(imgs, None))).T:
            imgs[i,j] = Spacer(imgs[i-1,j].get_box_width(),
                               imgs[i,j-1].get_box_height())
            
        # translate row hdr so that its bottom left corner is
        # on the bottom left corner of the image grid
        img_bot_y = row_header[-1][0].get_box_bot_y()
        dy = imgs[-1,0].get_box_bot_y() - img_bot_y
        row_hdr_group.translate(tx=0,ty=dy)
    
        # rotate row hdr by 90 deg. around the bottom left corner of the
        # image grid
        rot_ctr = (0, img_bot_y)
        row_hdr_group.rotate(-90, center=rot_ctr)    
        table_group.add(row_hdr_group)
        
        img_group = svgwrite.container.Group(id='table_content')
        for row_imgs in imgs:
            for img in row_imgs:
                img_group.add(img.to_svg())
        table_group.add(img_group)
    
        dwg.add(table_group)
        return dwg.tostring()


def dtree_to_table_elements(dtree, root_path, branch_names, row_branches,
                            column_branches):
    """
    Convert a data tree to a table of images with column and row headers

    TODO: doc, test, implement
    """
    # Reshape tree to match target row and column axes 
    dtree = tree.tree_rearrange(dtree, branch_names,
                                row_branches + column_branches)
    levels = tree.dtree_get_levels(dtree)

    row_levels = levels[:len(row_branches)]
    nb_rows = np.prod([len(lvl) for lvl in row_levels])
    col_levels = levels[len(row_branches):]
    nb_cols = np.prod([len(lvl) for lvl in col_levels])
    
    # Set image size relative to text size:
    img_h = BoxedText.DEFAULT_FONT_H * 7    
    img_it = tree.tree_leaves_iterator(dtree)
    
    def make_img(img_fn):
        return BoxedImage(op.join(root_path, img_fn), img_h=img_h)    
    bimgs = np.array([[make_img(img_it.next()) for icol in xrange(nb_cols)]\
                      for irow in xrange(nb_rows)], dtype=object)
    
    return row_levels, col_levels, bimgs
    
def dtree_to_svg(dtree, root_path, branch_names, row_branches,
                 column_branches, row_base_gap=Table.DEFAULT_ROW_BGAP,
                 col_base_gap=Table.DEFAULT_COL_BGAP):
    """
    Helper function to directly convert a dtree to a SVG string representation

    See 'dtree_to_table_elements' for doc on arguments
    """
    table = Table.from_dtree(dtree, root_path, branch_names,
                             row_branches, column_branches)
    table.add_spacers(row_base_gap=row_base_gap,
                      col_base_gap=col_base_gap)
    table.adjust_cell_sizes()
    return table.to_svg()
