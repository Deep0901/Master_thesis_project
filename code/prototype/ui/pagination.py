"""Pagination UI helpers for Streamlit results."""

from __future__ import annotations

from typing import Tuple

import streamlit as st


def render_pagination_controls(
    current_page: int,
    total_pages: int,
    key_suffix: str = "",
    show_page_jump: bool = True,
    compact: bool = False,
) -> Tuple[bool, bool, int]:
    """Render pagination controls (Previous/Next buttons and optional page jump).
    
    Args:
        current_page: Current page number (0-indexed).
        total_pages: Total number of pages.
        key_suffix: Suffix for button/selectbox keys (for uniqueness).
        show_page_jump: Whether to show page jump selector.
        compact: If True, use simpler layout (less whitespace).
    
    Returns:
        (should_go_prev: bool, should_go_next: bool, new_page_index: int)
    """
    should_go_prev = False
    should_go_next = False
    new_page_index = current_page

    if compact:
        pcol1, pcol2, pcol3, pcol4 = st.columns([1, 2, 2, 1])
    else:
        pcol1, pcol2, pcol3, pcol4 = st.columns([1, 2, 2, 1])

    with pcol1:
        if st.button(
            "Previous",
            disabled=(current_page <= 0),
            key=f"pag_prev_{key_suffix}",
            width="stretch",
        ):
            should_go_prev = True
            new_page_index = max(0, current_page - 1)

    with pcol2:
        st.markdown(
            f"<div style='text-align: center; padding-top: 8px;'><strong>Page {current_page + 1} of {total_pages}</strong></div>",
            unsafe_allow_html=True,
        )

    with pcol3:
        if show_page_jump and total_pages <= 20:
            page_options = list(range(1, total_pages + 1))
            selected_page = st.selectbox(
                "Jump to page",
                options=page_options,
                index=current_page,
                key=f"pag_jump_{key_suffix}",
                label_visibility="collapsed",
            )
            if selected_page - 1 != current_page:
                new_page_index = selected_page - 1
        else:
            st.markdown(" ")

    with pcol4:
        if st.button(
            "Next",
            disabled=(current_page >= total_pages - 1),
            key=f"pag_next_{key_suffix}",
            width="stretch",
        ):
            should_go_next = True
            new_page_index = min(total_pages - 1, current_page + 1)

    return should_go_prev, should_go_next, new_page_index


def render_pagination_summary(
    found_count: int,
    displayed_count: int,
    current_page: int,
    total_pages: int,
) -> None:
    """Render a summary line with pagination info.
    
    Example: "Found 1,234 datasets • Showing 20 results (page 1/62)"
    """
    st.markdown(
        f"### Found {found_count:,} datasets • Showing {displayed_count} results (page {current_page + 1}/{total_pages})"
    )
