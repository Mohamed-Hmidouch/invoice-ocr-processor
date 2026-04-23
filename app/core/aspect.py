"""
Aspect-Oriented Programming (AOP) — Cross-cutting concerns.

Ce module contient des décorateurs qui centralisent les préoccupations
transversales (logging, gestion d'erreurs) afin d'éviter la duplication
de blocs try/except dans chaque fichier.
"""
import logging
import functools
from typing import Type, Tuple

from app.core.exceptions import InvoiceProcessorError

logger = logging.getLogger(__name__)


def handle_exceptions(
    *catch: Type[Exception],
    raise_as: Type[InvoiceProcessorError] = InvoiceProcessorError,
    default_return=None,
    propagate: bool = True,
):
    """
    Décorateur AOP pour la gestion globale des exceptions.

    Paramètres
    ----------
    *catch : Type[Exception]
        Types d'exceptions à intercepter. Par défaut : Exception.
    raise_as : Type[InvoiceProcessorError]
        Type d'exception métier à relancer après interception.
    default_return :
        Valeur de retour en cas d'erreur si `propagate` est False.
    propagate : bool
        Si True, relance l'erreur encapsulée. Si False, retourne `default_return`.

    Exemple
    -------
    @handle_exceptions(FileNotFoundError, raise_as=ImageNotFoundError)
    def load_image(self, path):
        ...
    """
    if not catch:
        catch = (Exception,)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except raise_as:
                # Déjà une exception métier → on la laisse remonter telle quelle
                raise
            except catch as exc:
                logger.error(
                    "[%s] %s → %s: %s",
                    func.__qualname__,
                    type(exc).__name__,
                    raise_as.__name__,
                    exc,
                )
                if propagate:
                    raise raise_as(str(exc), detail=type(exc).__name__) from exc
                return default_return

        return wrapper

    return decorator
