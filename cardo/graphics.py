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
import logging
from itertools import chain

import numpy as np
import svgwrite
import base64

from cardo import tree

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

    def __repr__(self):
        return '(%d,%d)[%d,%d]' %(self.get_box_size()+ self.get_box_coords())
        
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

    def get_box_bot_right_coords(self):
        return self.get_box_right_x(), self.get_box_bot_y()
    
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
        return self.text + super(BoxedText, self).__repr__()

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
    def __init__(self, width=0, height=0, box_x=0, box_y=0):
        BoxedRect.__init__(self, width, height, box_x, box_y)

    def __repr__(self):
        return '@' + super(Spacer, self).__repr__()
       
    def to_svg(self):
        return svgwrite.shapes.Rect(insert=self.get_box_coords(),
                                    size=self.get_box_size(),
                                    style='stroke:none;fill:none')

class WrongSVGImgInclusion(Exception):
    pass
       
class BoxedImage(BoxedRect):

    IMG_EXT_REL = 'external_relative'
    IMG_EXT_ABS = 'external_absolute'
    IMG_EMBED = 'embedded'
    
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
        return op.basename(self.img_fn) + super(BoxedImage, self).__repr__()
        
    def to_svg(self, inclusion=IMG_EXT_ABS, ref_path=None):
        """
        Export image to SVG. Image can be linked as absolute or relative 
        file name, or image can be embedded.

        Args:
            - inclusion (str in (BoxedImage.IMG_EXT_REL, 
                                 BoxedImage.IMG_EXT_ABS,
                                 BoxedImage.IMG_EMBED):
              Inclusion option for the image:
                  - BoxedImage.IMG_EXT_REL: use an external link with a relative
                    path. *ref_path* MUST be provided.
                  - BoxedImage.IMG_EXT_ABS: use an external link with an absolute
                    path
                  - BoxedImage.IMG_EMBED: embed the content of the image file
                    in the return SVG string
            - ref_path (str):
              reference start path to set relative image file name. 
              Used when inclusion == BoxedImage.IMG_EXT_REL 

        Output: svgwrite.image.Image object
             SVG representation of the image
                
        """
        extra_attrs = {}
        if inclusion == BoxedImage.IMG_EXT_ABS:
            img_fn = op.abspath(self.img_fn)
        elif inclusion == BoxedImage.IMG_EXT_REL:
            assert ref_path is not None
            img_fn = op.relpath(self.img_fn, ref_path)
            extra_attrs = {'sodipodi:xlink:absref' : op.abspath(self.img_fn)}
        elif inclusion == BoxedImage.IMG_EMBED:
            formats = {'.jpg':'jpeg', '.jpeg':'jpeg', '.png':'png'}
            ext = op.splitext(self.img_fn)[-1]
            if ext in formats:
                img_fn = 'data:image/%s;base64,%s' \
                         %(formats[ext], encode64_img(self.img_fn))
        else:
            raise WrongSVGImgInclusion('Wrong inclusion type %s' % inclusion)
        return svgwrite.image.Image(img_fn, insert=self.get_rect_coords(),
                                    size=(self.rect_width, self.rect_height),
                                    debug=False, **extra_attrs)

def encode64_img(img_fn):
    with open(img_fn) as fimg:
        encoded_img = base64.b64encode(fimg.read())
    return encoded_img

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
        return '.' + self.gfx_element.__repr__()

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
        def _breath_first_walk(node):
            visited, queue = set(), [(self,-1)]
            sizes = [0]
            cur_level = -1
            while queue:
                node, level = queue.pop(0)
                if level != cur_level:
                    sizes.append(0)
                    cur_level = level
                if node.gfx_element is not None:
                        sizes[-1] = max(sizes[-1], len(str(node.gfx_element)))
                if node not in visited:
                    visited.add(node)
                    queue.extend((c, cur_level+1) for c in node.children \
                                 if c not in visited)
            return sizes
        level_sizes = _breath_first_walk(self)
            
        def _depth_first_walk(node, line, ilevel):
            node_str = str(node.gfx_element).ljust(level_sizes[ilevel])
            if node.is_leaf():
                print line + ' | ' + node_str
            else:
                for child in node.children:
                    _depth_first_walk(child, line + ' | ' + node_str, ilevel+1)
        _depth_first_walk(self, '', 0)

        
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
            - images (numpy array of BoxedImage):
                 2d array of image elements. image.shape[0] must equal
                 the cartesian product of hdr_levels
        """
        def leaf_img_trees():
            for irow in xrange(len(images)):
                col_root = GTree(images[irow][0])
                cur_node = col_root
                for icol in xrange(1, len(images[0])):
                    next_child = GTree(images[irow][icol], parent=cur_node)
                    cur_node.add_child(next_child)
                    cur_node = next_child
                yield col_root
        img_tree_iterator = leaf_img_trees()
        
        
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
                    img_tree = img_tree_iterator.next()
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
            logger.debug('_space_rec: height=%d, node: %s',
                         gap_factor, str(node))
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
                                    if not isinstance(s.gfx_element, Spacer)) \
                                * 1.
                logger.debug('nb non-spacers: %d', nb_non_spacer)
                spacer_size = sum(get_size(sib.gfx_element) \
                                  for isib, sib in enumerate(siblings) \
                                  if isinstance(sib.gfx_element, Spacer))
                logger.debug('Gap size: %d', spacer_size)
                par_based_size = math.floor((get_size(self.parent.gfx_element) -\
                                             spacer_size) /                     \
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
                     leveled=True, max_depth=-1, reverse_child_order=False):
        """
        Gather all elements that match the given predicator *filt*.
        Do a breath-first or depth-first walk of the tree.
        
        Output: 
            - If *leveled* is True then:
                 list of list of BoxedElements
                 - first dimension is parsed layers
                 - second dimension is elements in the current layer 
            - If *leveled* is False then:
                 list of BoxedElements

        TODO: test max_depth, reverse
        """
        if filt is None:
            filt = lambda x: True

        if reverse_child_order:
            parse_child = lambda x: reversed(x)
        else:
            parse_child = lambda x: x
            
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
                    if leveled and level >= start_depth:
                        elements.append([])
                    cur_level = level
                if level >= start_depth and node.gfx_element is not None and \
                   filt(node.gfx_element):
                    if leveled:
                        elements[-1].append(node.gfx_element)
                    else:
                        elements.append(node.gfx_element)
                if node not in visited:
                    visited.add(node)
                    queue.extend((c, cur_level+1)
                                 for c in parse_child(node.children) \
                                 if c not in visited)
        else: # detph-1st-search
            elements = []
            def _depth_first_walk(node, depth):
                logger.debug('dfw, node: %s, depth=%d', str(node), depth)
                if node.is_leaf() or (max_depth != -1 and depth == max_depth):
                    if filt(node.gfx_element) and depth >= start_depth:
                        elements.append(node.gfx_element)
                else:
                    if node.gfx_element is None: # skip tree root
                        for child in parse_child(node.children):
                            _depth_first_walk(child, depth)
                    else:
                        for child in parse_child(node.children):
                            if filt(node.gfx_element) and depth >= start_depth:
                                elements.append(node.gfx_element)
                            _depth_first_walk(child, depth + 1)
            
            def _depth_first_walk_leveled(node, branch, depth):
                logger.debug('dfw, pending branch: %s', str(branch))
                logger.debug('dfw, current node: %s, depth=%d', str(node), depth)
                if node.is_leaf() or (max_depth != -1 and depth >= max_depth):
                    logger.debug('Arrived to leaf, append content and return')
                    if filt(node.gfx_element) and depth >= start_depth and \
                       (depth < max_depth or max_depth == -1):
                        branch = branch + [node.gfx_element]
                    else:
                        logger.debug('Gfx element does not match filter or '\
                                     'max_depth reached')
                    elements.append(branch)
                    logger.debug('Updated walked elements: %s', str(elements))
                else:
                    if node.gfx_element is None: # skip tree root
                        for child in parse_child(node.children):
                            _depth_first_walk_leveled(child, branch, depth)
                    else:
                        for child in parse_child(node.children):
                            if depth >= start_depth and filt(node.gfx_element) :
                                _depth_first_walk_leveled(child, branch + \
                                                          [node.gfx_element],
                                                          depth + 1)
                            else:
                                _depth_first_walk_leveled(child, branch,
                                                          depth + 1)
            if not leveled:
                _depth_first_walk(self, 0)
            else:
                _depth_first_walk_leveled(self, [], 0)
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
                      - row_hdr_levels[0][0], row_hdr_levels[1][0],  ...
                        row_hdr_levels[-1][0]
                -> the cartesian product of col_hdr_levels has the same size as
                   the size of the 1st row of images
                -> the cartesian product of row_hdr_levels has the same size as
                   the size of the 1st column of images                
        Args:
            - row_hdr_levels (list of list of str):
                texts for each levels in the row header
            - col_hdr_levels (list of list of str):
                texts for each levels in the column header
            - imgages (numpy array of BoxedImage):
                2d array of images
        """
        assert(np.prod([len(lvl) for lvl in col_hdr_levels]) == len(images[0]))
        assert(np.prod([len(lvl) for lvl in row_hdr_levels]) == len(images))

        self.col_hdr_height = len(col_hdr_levels)
        self.col_gtree = GTree.from_hdr_and_images(col_hdr_levels, images.T)

        self.row_hdr_height = len(row_hdr_levels)
        self.row_gtree = GTree.from_hdr_and_images(row_hdr_levels, images)

        self.images = images
        logger.debug('Init Table with row hdr levels: %s, col hdr levels: %s' \
                     ', grid image of size %s -> built a row hdr tree of '\
                     'height %d and a col hdr tree of height %d.',
                     str(row_hdr_levels), str(col_hdr_levels), str(images.shape),
                     self.row_gtree.get_height(), self.col_gtree.get_height())
                             
    @staticmethod
    def from_dtree(dtree, branch_names, row_branches,
                   column_branches):
        return Table(*dtree_to_table_elements(dtree, branch_names,
                                              row_branches, column_branches))

    def get_col_header_elements(self):
        return self.col_gtree.get_elements(max_depth=self.col_hdr_height)

    def get_row_header_elements(self):
        return self.row_gtree.get_elements(max_depth=self.row_hdr_height)
    
    def add_spacers(self, row_base_gap=DEFAULT_ROW_BGAP,
                    col_base_gap=DEFAULT_COL_BGAP):

        def create_col_spacer(gap, ref_sibling):
            return Spacer(width=gap, height=ref_sibling.get_box_height())
        self.col_gtree.add_spacers(col_base_gap, create_col_spacer,
                                   self.col_hdr_height-1)
        
        def create_row_spacer(gap, ref_sibling):
            if isinstance(ref_sibling, BoxedImage):
                return Spacer(height=gap, width=ref_sibling.get_box_height())
            else:
                return Spacer(width=gap, height=ref_sibling.get_box_height())    
        self.row_gtree.add_spacers(row_base_gap, create_row_spacer,
                                   self.row_hdr_height-1)
            
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
        self.row_gtree.adjust_size(rowgt_get_size, rowgt_set_size)

    def get_elements_via_column_header(self, walk_type='bfs'):
        return self.col_gtree.get_elements(walk_type=walk_type)

    def get_elements_via_row_header(self, walk_type='bfs'):
        return self.row_gtree.get_elements(walk_type=walk_type)

    
    def deoverlap(self):
        """
        Move all elements to remove all overlaps.
        For headers, elements are moved from left to right
        For the grid of images, elements are moved from left to right and
        from top to bottom
        TODO: test
        """
        logger.debug('Deoverlap table ...')

        logger.debug('vDeoverlap elements from col view ...')
        vlines_via_chdr = self.col_gtree.get_elements(walk_type='dfs')
        for vline in vlines_via_chdr:
            logger.debug('Deoverlap vline: %s ...', str(vline))
            BoxedElement.vdeoverlap(vline)
            logger.debug('->  %s', str(vline))

        logger.debug('hDeoverlap elements from col view ...')            
        hlines_via_chdr = self.col_gtree.get_elements(max_depth=self.col_hdr_height,
                                                      walk_type='bfs')
        for hline in hlines_via_chdr:
            logger.debug('Deoverlap hline: %s ...', str(hline))
            BoxedElement.hdeoverlap(hline)
            logger.debug('->  %s', str(hline))

        logger.debug('vDeoverlap elements from row view ...')
        vlines_via_rhdr = self.row_gtree.get_elements(max_depth=self.row_hdr_height,
                                                      walk_type='dfs')
        for vline in vlines_via_rhdr:
            logger.debug('Deoverlap vline: %s ...', str(vline))
            BoxedElement.vdeoverlap(vline)
            logger.debug('->  %s', str(vline))
        
        logger.debug('hDeoverlap elements from row view (reversed!) ...')
        hlines_via_rhdr = self.row_gtree.get_elements(max_depth=self.row_hdr_height,
                                                      walk_type='bfs')
        for hline in hlines_via_rhdr:
            logger.debug('Deoverlap hline: %s ...', str(vline))
            BoxedElement.hdeoverlap(list(reversed(hline)))
            logger.debug('->  %s', str(vline))

        logger.debug('hDeoverlap img elements from row view ...')            
        row_imgs = self.row_gtree.get_elements(start_depth=self.row_hdr_height,
                                               walk_type='dfs')
        for row in row_imgs:
            logger.debug('Deoverlap img row: %s ...', str(row))
            BoxedElement.hdeoverlap(row)
            logger.debug('->  %s', str(row))
                   
    @staticmethod
    def is_hdr_element(element):
        return not isinstance(element, BoxedImage)

    @staticmethod
    def is_img_element(element):
        return isinstance(element, BoxedImage)


    def get_table_parts(self):
        """
        
        Outputs are not leveled
        """
        col_header = self.col_gtree.get_elements(max_depth=self.col_hdr_height)
        row_header = self.row_gtree.get_elements(max_depth=self.row_hdr_height)

        imgs_cview = self.col_gtree.get_elements(start_depth=self.col_hdr_height)
        if not isinstance(imgs_cview[0][1], Spacer):
            imgs = np.array(imgs_cview)
        else:
            rh = self.row_hdr_height
            imgs_rview = self.row_gtree.get_elements(start_depth=rh,
                                                     walk_type='dfs')
            imgs = np.empty((len(row_header[-1]), len(col_header[-1])),
                            dtype=object)
            imgs[::2, :] = np.array(imgs_cview, dtype=object)
            imgs[1::2, ::2] = np.array(imgs_rview, dtype=object)[1::2, :]
            for i, j in np.vstack(np.where(np.equal(imgs, None))).T:
                imgs[i, j] = Spacer(imgs[i-1, j].get_box_width(),
                                    imgs[i, j-1].get_box_height(),
                                    imgs[i, j-1].get_box_right_x(),
                                    imgs[i-1, j].get_box_bot_y())

            for row in imgs:
                BoxedElement.hdeoverlap(row)
                
            for col in imgs.T:
                BoxedElement.vdeoverlap(col)
                
        return list(chain(*row_header)), list(chain(*col_header)), \
            list(imgs.flatten())
        
    def to_svg(self, ref_path=None):

        row_header, column_header, grid = self.get_table_parts()
        
        dwg = svgwrite.Drawing()
        table_group = svgwrite.container.Group(id='table')
        
        if len(column_header) > 0:
            col_hdr_group = svgwrite.container.Group(id='table_col_hdr')
            for element in column_header:
                col_hdr_group.add(element.to_svg())            
            table_group.add(col_hdr_group)

        if len(row_header) > 0:
            row_hdr_group = svgwrite.container.Group(id='table_row_hdr')
            for element in row_header:
                row_hdr_group.add(element.to_svg())
            # translate row hdr so that its bottom left corner is
            # on the bottom left corner of the image grid
            hdr_bot_y = row_header[-1].get_box_bot_y()
            dy = grid[-1].get_box_bot_y() - hdr_bot_y
            row_hdr_group.translate(tx=0,ty=dy)
            # rotate row hdr by 90 deg. around the bottom left corner of the
            # image grid
            rot_ctr = (0, hdr_bot_y)
            row_hdr_group.rotate(-90, center=rot_ctr)    
            table_group.add(row_hdr_group)
            
        content_group = svgwrite.container.Group(id='table_content')
        if ref_path is not None:
            img_inclusion = BoxedImage.IMG_EXT_REL
        else:
            img_inclusion = BoxedImage.IMG_EXT_ABS
        # TODO: handle embedded images
        for grid_elem in grid:
            if isinstance(grid_elem, BoxedImage):
                content_group.add(grid_elem.to_svg(img_inclusion, ref_path))
            else:
                content_group.add(grid_elem.to_svg())
        table_group.add(content_group)
    
        dwg.add(table_group)
        return dwg


def dtree_to_table_elements(dtree, branch_names, row_branches,
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
        return BoxedImage(img_fn, img_h=img_h)    
    bimgs = np.array([[make_img(img_it.next()) for icol in xrange(nb_cols)]\
                      for irow in xrange(nb_rows)], dtype=object)
    
    return row_levels, col_levels, bimgs
    
def dtree_to_svg(dtree, branch_names, row_branches,
                 column_branches, row_base_gap=Table.DEFAULT_ROW_BGAP,
                 col_base_gap=Table.DEFAULT_COL_BGAP, ref_path=None):
    """
    Helper function to directly convert a dtree to a SVG string representation

    TODO: add option to embed images in SVG doc
    """
    table = Table.from_dtree(dtree, branch_names, row_branches, column_branches)
    table.add_spacers(row_base_gap=row_base_gap,
                      col_base_gap=col_base_gap)
    table.adjust_cell_sizes()
    table.deoverlap()
    return table.to_svg(ref_path=ref_path)
