import PhotoSwipe, { Options } from 'photoswipe';
import { Item, ItemOptions } from '../Item';
/**
 * A map of all possible event and the structure of their details
 */
export interface CustomEventDetailMap<T> {
    'activate': {
        model: T;
        clickEvent: MouseEvent;
    };
    'item-added-to-dom': T;
    'item-displayed': T;
    'pagination': {
        offset: number;
        limit: number;
    };
    'select': T[];
    'zoom': {
        item: T;
        photoswipe: PhotoSwipe<Options>;
    };
}
/**
 * Augment the global namespace with our custom events
 * See: https://github.com/Microsoft/TypeScript/issues/28357
 */
declare global {
    interface HTMLElementEventMap {
        activate: CustomEvent;
        'item-added-to-dom': CustomEvent;
        'item-displayed': CustomEvent;
        pagination: CustomEvent;
        select: CustomEvent;
        zoom: CustomEvent;
    }
}
export interface SizedModel {
    /**
     * Height in pixels of the enlarged version the image
     * If photoswipe is used, the size of the photoswipe enlarged image is required
     * If photoswipe is not used, any size that match the ratio is enough
     */
    enlargedWidth: number;
    /**
     * Width in pixels of the enlarged version the image
     * If photoswipe is used, the size of the photoswipe enlarged image is required
     * If photoswipe is not used, any size that match the ratio is enough
     */
    enlargedHeight: number;
}
export interface ModelAttributes extends SizedModel {
    /**
     * Source link for thumbnail image
     */
    thumbnailSrc: string;
    /**
     * Source link for enlarged (photoswipe) image
     */
    enlargedSrc?: string;
    /**
     * Label of item (or button)
     */
    title?: string;
    /**
     * Href link
     */
    link?: string;
    /**
     * a href target attribute
     */
    linkTarget?: '_blank' | '_self' | '_parent' | '_top';
    /**
     * Hex color
     */
    color?: string;
    /**
     * If item is selected
     */
    selected?: boolean;
    /**
     * Background size, default : cover
     */
    backgroundSize?: string;
    /**
     * Background position, default : center
     */
    backgroundPosition?: string;
}
export interface GalleryOptions extends ItemOptions {
    rowsPerPage?: number;
    minRowsAtStart?: number;
    infiniteScrollOffset?: number;
    photoSwipeOptions?: PhotoSwipeOptions | null;
    ssr?: {
        /**
         * In SSR mode, if the gallery width cannot be computed, it will fallback to this value
         */
        galleryWidth: number;
    };
}
declare type PhotoSwipeOptions = Options;
export interface InnerPhotoSwipeOptions extends PhotoSwipeOptions {
    index: number;
    loop: boolean;
}
export interface PhotoswipeItem {
    src?: string;
    w: number;
    h: number;
    title?: string;
}
export declare abstract class AbstractGallery<Model extends ModelAttributes> {
    protected elementRef: HTMLElement;
    protected photoswipeElementRef?: HTMLElement | null | undefined;
    protected scrollElementRef?: HTMLElement | null | undefined;
    /**
     * Default options
     */
    protected options: Required<GalleryOptions>;
    protected photoswipeDefaultOptions: PhotoSwipeOptions;
    /**
     * Images wrapper container
     * If setted, serves as mark for "initialized status" of the gallery
     */
    protected bodyElementRef: HTMLElement | null;
    /**
     * Items for which container has been added to dom, but image has not been queries yet
     */
    protected scrollBufferedItems: Item<Model>[];
    /**
     * Debounce function
     * Runs a small delay after last image has been added to dom
     * When it runs, images are loaded (appear with fade) and more images are queries to preserve a buffer of
     * out-of-dom items
     */
    protected flushBufferedItems: () => void;
    /**
     * Number of items to query on buffer flushing
     */
    protected requiredItems: number;
    /**
     * Used to test the scroll direction
     * Avoid to load more images when scrolling up
     */
    private old_scroll_top;
    /**
     * Photoswipe images container
     * @type {Array}
     */
    private photoswipeCollection;
    /**
     * Reference to next button element
     */
    private nextButton;
    protected readonly document: Document;
    /**
     *
     * @param elementRef
     * @param options
     * @param photoswipeElementRef
     * @param scrollElementRef
     */
    constructor(elementRef: HTMLElement, options: GalleryOptions, photoswipeElementRef?: HTMLElement | null | undefined, scrollElementRef?: HTMLElement | null | undefined);
    /**
     * Complete collection of images
     * @type {Array}
     */
    protected _collection: Item<Model>[];
    get collection(): Item<Model>[];
    /**
     * Partial set of items that represent the visible items
     * @type {Item[]}
     * @private
     */
    protected _visibleCollection: Item<Model>[];
    get visibleCollection(): Item<Model>[];
    get selectedItems(): Model[];
    get width(): number;
    get collectionLength(): number;
    get visibleCollectionLength(): number;
    /**
     * Initializes DOM manipulations
     */
    init(): void;
    /**
     * Add items to collection
     * Transform given list of models into inner Items
     * @param models list of models
     */
    addItems(models: Model[]): void;
    /**
     * Select all items visible in the DOM
     * Ignores buffered items
     */
    selectVisibleItems(): Model[];
    /**
     * Unselect all selected elements
     */
    unselectAllItems(): void;
    /**
     * Allows to use the same approach and method name to listen as gallery events on DOM or on javascript gallery
     * object
     *
     * Gallery requests items when it's instantiated. But user may subscribe after creation, so we need to request
     * again if user subscribes by this function.
     *
     * @param name
     * @param callback
     */
    addEventListener<K extends keyof CustomEventDetailMap<Model>>(name: K, callback: (evt: CustomEvent<CustomEventDetailMap<Model>[K]>) => void): void;
    /**
     * Public api for empty function
     * Emits a pagination event
     */
    clear(): void;
    /**
     * Return copy of options to prevent modification
     */
    getOptions(): GalleryOptions;
    /**
     * Override current collection
     * @param {Item[]} items
     */
    setItems(items: Model[]): void;
    /**
     *
     */
    abstract organizeItems(items: Item<Model>[], fromRow?: number, toRow?: number): void;
    /**
     * If gallery already has items on initialisation, set first page visible, load second page and query for more
     * items if needed If not, just query for items
     */
    protected initItems(): void;
    /**
     *
     */
    protected abstract getEstimatedColumnsPerRow(): number;
    /**
     * AbstractRowGallery + Masonry
     */
    protected abstract onScroll(): void;
    /**
     * AbstractRowGallery + Masonry
     */
    protected abstract onPageAdd(): void;
    /**
     * Return number of rows to show per page to fill the empty space until the bottom of the screen
     * Should grant all the space is used or more, but not less.
     * @returns {number}
     */
    protected abstract getEstimatedRowsPerPage(): number;
    /**
     * Fire pagination event
     * Information provided in the event allows to retrieve items from the server using given data :
     * "offset" and "limit" that have the same semantic that respective attributes in mySQL.
     *
     * The gallery asks for items it needs, including some buffer items that are not displayed when given but are
     * available to be added immediately to DOM when user scrolls.
     *
     */
    protected requestItems(): void;
    /**
     * Returns option.rowsPerPage is specified.
     * If not returns the estimated number of rows to fill the rest of the vertical space in the screen
     * @returns {number}
     */
    protected getRowsPerPage(): number;
    /**
     * Add given item to DOM and to visibleCollection
     * @param {Item} item
     * @param destination
     */
    protected addItemToDOM(item: Item<Model>, destination?: HTMLElement | null): void;
    protected updateNextButtonVisibility(): void;
    /**
     * If infinite scroll (no option.rowsPerPage provided), a minimum height is setted to force gallery to overflow
     * from viewport. This activates the scroll before adding items to dom. This prevents the scroll to fire new resize
     * event and recompute all gallery twice on start.
     */
    protected extendToFreeViewport(): void;
    /**
     * Space between the top of the gallery wrapper (parent of gallery root elementRef) and the bottom of the window
     */
    protected getGalleryVisibleHeight(): number;
    protected startResize(): void;
    protected endResize(): void;
    protected openPhotoSwipe(item: Item<Model>): void;
    /**
     * Format an Item into a PhotoswipeItem that has different attributes
     * @param item
     * @returns {PhotoswipeItem}
     */
    protected getPhotoswipeItem(item: Item<Model>): PhotoswipeItem;
    protected dispatchEvent<K extends keyof CustomEventDetailMap<Model>>(name: K, data: CustomEventDetailMap<Model>[K]): void;
    /**
     * Effectively empty gallery, and should prepare container to receive new items
     */
    protected empty(): void;
    /**
     * Listen to scroll event and manages rows additions for lazy load
     * @param {HTMLElement | Document} element
     */
    private bindScroll;
}
export {};
