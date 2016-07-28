import os.path as op
import imghdr
import struct
import math
from pprint import pformat
import logging

import svgwrite

logger = logging.getLogger('cardo')

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
        return self.text

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
               

# class SizeTree(object):

#     sizes = {}
   
#     def __init__(self, level, size=0., children=None, parent=None):
#         self.parent = parent
#         self.level = level
       
#         self.update_level_size(size)

#         self.children = children
#         if self.children is None:
#             self.children = []

#         if self.parent is not None:
#             self.parent.add_child(self)
#             assert parent.get_level() == level - 1
           
#         for child in self.children:
#             child.set_parent(self)
#             assert child.get_level() == level + 1

#     @staticmethod
#     def reset_sizes():
#         SizeTree.sizes = {}
       
#     def get_level(self):
#         return self.level
           
#     def __repr__(self):
#         return '[l' + str(self.level) + 's' + str(self.get_size()) + ']'

#     def add_child(self, c):
#         self.children.append(c)
       
#     def set_parent(self, p):
#         self.parent = p
           
#     def get_size(self):
#         return SizeTree.sizes[self.level]

#     def set_size(self, size):
#         SizeTree.sizes[self.level] = size

#     def update_level_size(self, size):
#         if SizeTree.sizes.get(self.level, 0) < size:
#             SizeTree.sizes[self.level] = size
       
#     def get_children(self):
#         return self.children

#     def adjust(self, parent_size=0):
#         nchilds = len(self.children)
#         child_extent = sum([c.adjust(self.get_size()/nchilds) \
#                             for c in self.children])
#         self.set_size(max((child_extent, parent_size, self.get_size())))
#         return self.get_size()
       
    # def adjust_to_siblings(self):
    #     """ Adjust sizes via breadth first search """
    #     def _bfs(tree, level):
    #         yield (tree, level)
    #         last = tree
    #         for sibling,level in _bfs(tree, level+1):
    #             for child in sibling.get_children():
    #                 yield (child, level)
    #                 last = child
    #             if last == sibling:
    #                 return

    #     # print 'bsf:', list(_bfs(self, 0))

    #     cur_level=0
    #     group=[]
    #     for node,level in _bfs(self, 0):
    #         if level != cur_level: #jumped to a new level
    #             # resolve level parsed so far
    #             adjusted_size = max([n.get_size() for n in group])
    #             for n in group:
    #                 n.set_size(adjusted_size)
    #             # reset group & go to the next level
    #             cur_level = level
    #             group[:] = []

    #         group.append(node)
    #     # do last group:
    #     adjusted_size = max([n.get_size() for n in group])
    #     for n in group:
    #         n.set_size(adjusted_size)
           
    #     # print 'bsf:', list(_bfs(self, 0))
